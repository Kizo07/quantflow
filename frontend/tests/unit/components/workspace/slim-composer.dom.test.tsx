import { afterEach, describe, expect, rs, test } from "@rstest/core";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import type { ReactNode } from "react";

import { SlimComposer } from "@/components/workspace/slim-composer";
import type { Locale } from "@/core/i18n";
import { I18nProvider } from "@/core/i18n/context";

afterEach(() => {
  cleanup();
});

function renderSlim(node: ReactNode, locale: Locale = "en-US") {
  document.cookie = `locale=${locale}; path=/`;
  return render(
    <I18nProvider initialLocale={locale}>{node}</I18nProvider>,
  );
}

// jsdom does not turn a submit-button click into a form submit, so these
// drive the form directly (the same path Enter and the button take).
// `PromptInput` delivers the message asynchronously (file conversion),
// hence the `waitFor` on the callbacks.

describe("SlimComposer", () => {
  test("sends typed text as a PromptInputMessage and clears the input", async () => {
    const onSubmit = rs.fn();
    const { container } = renderSlim(
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
    const { container } = renderSlim(
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
    const { container } = renderSlim(
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
    const { container } = renderSlim(
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

  test("always renders the shared disclaimer below the composer", () => {
    renderSlim(<SlimComposer onSubmit={rs.fn()} />);
    // Same copy as the full InputBox welcome disclaimer.
    expect(
      screen.getByText("QuantFlow is AI and can make mistakes"),
    ).not.toBeNull();
  });

  test("shows shared starter chips only in welcome mode", async () => {
    const { rerender } = renderSlim(
      <SlimComposer onSubmit={rs.fn()} placeholder="Ask the bot" />,
    );
    expect(
      document.querySelector('[data-slot="suggestions-list"]'),
    ).toBeNull();

    rerender(
      <I18nProvider initialLocale="en-US">
        <SlimComposer
          onSubmit={rs.fn()}
          isWelcomeMode
          placeholder="Ask the bot"
        />
      </I18nProvider>,
    );
    const starters = await screen.findByRole("button", { name: "Write" });
    expect(starters).not.toBeNull();
    // Research-only chrome stays out of the slim composer.
    expect(screen.queryByRole("button", { name: "Surprise" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Create" })).toBeNull();

    const textarea = screen.getByPlaceholderText("Ask the bot");
    fireEvent.click(starters);
    await waitFor(() =>
      expect((textarea as HTMLTextAreaElement).value).toContain(
        "Write a blog post",
      ),
    );
  });
});
