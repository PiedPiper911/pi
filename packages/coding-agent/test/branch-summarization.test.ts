import type { StreamFn } from "@earendil-works/pi-agent-core";
import {
	type AssistantMessage,
	createAssistantMessageEventStream,
	fauxAssistantMessage,
	type Model,
	type SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import { generateBranchSummary } from "../src/core/compaction/index.ts";
import type { SessionEntry } from "../src/core/session-manager.ts";

const model: Model<"anthropic-messages"> = {
	id: "test-model",
	name: "Test Model",
	api: "anthropic-messages",
	provider: "anthropic",
	baseUrl: "https://api.anthropic.com",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 200000,
	maxTokens: 8192,
};

const entries: SessionEntry[] = [
	{
		type: "message",
		id: "branch-user",
		parentId: null,
		timestamp: new Date(1).toISOString(),
		message: { role: "user", content: "Abandoned request", timestamp: 1 },
	},
];

function response(content: AssistantMessage["content"]): AssistantMessage {
	return {
		...fauxAssistantMessage(""),
		content,
		api: model.api,
		provider: model.provider,
		model: model.id,
	};
}

/** Captures the stream options the summarization request was issued with. */
function captureStreamFn(): {
	streamFn: StreamFn;
	options: () => SimpleStreamOptions | undefined;
} {
	let captured: SimpleStreamOptions | undefined;
	const streamFn: StreamFn = (_model, _context, options) => {
		captured = options;
		const stream = createAssistantMessageEventStream();
		queueMicrotask(() =>
			stream.push({ type: "done", reason: "stop", message: response([{ type: "text", text: "summary" }]) }),
		);
		return stream;
	};
	return { streamFn, options: () => captured };
}

describe("branch summarization", () => {
	it("does not override tool choice for branch summaries", async () => {
		let requestOptions: SimpleStreamOptions | undefined;
		const streamFn: StreamFn = (_model, _context, options) => {
			requestOptions = options;
			const stream = createAssistantMessageEventStream();
			queueMicrotask(() =>
				stream.push({ type: "done", reason: "stop", message: response([{ type: "text", text: "summary" }]) }),
			);
			return stream;
		};

		await generateBranchSummary(entries, {
			model,
			signal: new AbortController().signal,
			streamFn,
		});

		expect(requestOptions?.toolChoice).toBeUndefined();
	});

	it("rejects tool calls from branch summaries", async () => {
		const streamFn: StreamFn = () => {
			const stream = createAssistantMessageEventStream();
			queueMicrotask(() =>
				stream.push({
					type: "done",
					reason: "toolUse",
					message: response([
						{ type: "toolCall", id: "tool-call-1", name: "read", arguments: { path: "README.md" } },
					]),
				}),
			);
			return stream;
		};

		const result = await generateBranchSummary(entries, {
			model,
			signal: new AbortController().signal,
			streamFn,
		});

		expect(result.error).toBe("Branch summarization attempted to call a tool");
	});

	it("rejects length-limited branch summaries", async () => {
		const streamFn: StreamFn = () => {
			const stream = createAssistantMessageEventStream();
			queueMicrotask(() =>
				stream.push({
					type: "done",
					reason: "length",
					message: { ...response([{ type: "text", text: "partial" }]), stopReason: "length" },
				}),
			);
			return stream;
		};

		const result = await generateBranchSummary(entries, {
			model,
			signal: new AbortController().signal,
			streamFn,
		});

		expect(result.error).toBe(
			"Branch summarization failed: generation hit the token cap and the summary is incomplete",
		);
	});

	it("caps the output budget at the model limit when that is the tighter bound", async () => {
		const { streamFn, options } = captureStreamFn();

		await generateBranchSummary(entries, {
			model,
			signal: new AbortController().signal,
			streamFn,
		});

		// floor(0.8 * 16384) = 13107, but model.maxTokens is 8192
		expect(options()?.maxTokens).toBe(8192);
	});

	/** A 2048-token cap is smaller than most reasoning output, so large branches always hit it (#8845) */
	it("derives the output budget from reserveTokens instead of pinning it at 2048", async () => {
		const largeModel = { ...model, maxTokens: 65536 };
		const { streamFn, options } = captureStreamFn();

		await generateBranchSummary(entries, {
			model: largeModel,
			signal: new AbortController().signal,
			streamFn,
		});

		expect(options()?.maxTokens).toBe(13107);
		expect(options()?.maxTokens).toBeGreaterThan(2048);
	});

	it("scales the output budget with a custom reserveTokens", async () => {
		const largeModel = { ...model, maxTokens: 65536 };
		const { streamFn, options } = captureStreamFn();

		await generateBranchSummary(entries, {
			model: largeModel,
			reserveTokens: 32768,
			signal: new AbortController().signal,
			streamFn,
		});

		// floor(0.8 * 32768) = 26214
		expect(options()?.maxTokens).toBe(26214);
	});
});
