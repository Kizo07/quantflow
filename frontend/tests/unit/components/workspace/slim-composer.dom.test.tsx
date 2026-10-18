import { afterEach, describe, expect, rs, test } from "@rstest/core";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";

import { SlimComposer } from "@/components/workspace/slim-composer";

afterEach(() => {
  cleanup();
});

// jsdom does not turn a submit-button click into a form submit, so these
// drive the form directly (the same path Enter and the button take).
// `PromptInput` delivers the message asynchronously (file conversion),
// hence the `waitFor` on the callbacks.

describe("SlimComposer", () => {
  test("sends typed text as a PromptInputMessage and clears the input", async () => {
    const onSubmit = rs.fn();
    const { container } = render(
      <SlimComposer onSubmit={onSubmit} placeholder="Ask the bot" />,
    );

    const textarea = screen.getByPlaceholderText("Ask the bot");
    fireEvent.change(textarea, { target: { value: "Summarize today" } });
    fireEvent.submit(container.querySelector("form")!);

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0]![0]).toEqual({
      text: "Summarize today",
      files: [],
    });
    await waitFor(() =>
      expect((textarea as HTMLTextAreaElement).value).toBe(""),
    );
  });

  test("does not send empty text and keeps the submit button disabled", async () => {
    const onSubmit = rs.fn();
    const { container } = render(
      <SlimComposer onSubmit={onSubmit} placeholder="Ask the bot" />,
    );
    const submit = screen.getByRole("button", {
      name: "Submit",
    });
    expect(submit.hasAttribute("disabled")).toBe(true);
    fireEvent.submit(container.querySelector("form")!);
    // Give the async submit pipeline a chance to (not) deliver.
    await Promise.resolve();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  test("denies send without runs:create, like InputBox", async () => {
    const onSubmit = rs.fn();
    const { container } = render(
      <SlimComposer
        onSubmit={onSubmit}
        canCreateRuns={false}
        placeholder="Ask the bot"
      />,
    );
    const textarea = screen.getByPlaceholderText("Ask the bot");
    fireEvent.change(textarea, { target: { value: "Summarize today" } });
    const submit = screen.getByRole("button", { name: "Submit" });
    expect(submit.hasAttribute("disabled")).toBe(true);
    fireEvent.submit(container.querySelector("form")!);
    await Promise.resolve();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  test("stops instead of sending while streaming", async () => {
    const onSubmit = rs.fn();
    const onStop = rs.fn();
    const { container } = render(
      <SlimComposer
        onSubmit={onSubmit}
        onStop={onStop}
        status="streaming"
        placeholder="Ask the bot"
      />,
    );
    fireEvent.submit(container.querySelector("form")!);
    await waitFor(() => expect(onStop).toHaveBeenCalledTimes(1));
    expect(onSubmit).not.toHaveBeenCalled();
  });
});
