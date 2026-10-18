import { afterEach, describe, expect, test } from "@rstest/core";
import { cleanup, render, screen } from "@testing-library/react";

import { ScheduledOriginDivider } from "@/components/workspace/messages/scheduled-run-prompt";
import { I18nProvider } from "@/core/i18n/context";
import { scheduledOriginOf } from "@/core/messages/utils";
import type { ScheduledOrigin } from "@/core/scheduled-tasks/types";

afterEach(() => {
  cleanup();
  document.cookie = "locale=; max-age=0; path=/";
});

const ORIGIN: ScheduledOrigin = {
  task_id: "persistent-bot-heartbeat",
  task_run_id: "run-1",
  trigger: "scheduled",
  run_number: 3,
  scheduled_for: "2026-10-07T01:00:00+00:00",
  timezone: "UTC",
  schedule_type: "interval",
  task_title: "Persistent bot heartbeat",
  instructions: "Heartbeat check-in",
  stop_condition: null,
  standing_notes: [],
};

function renderDivider(origin: ScheduledOrigin, locale: "en-US" | "zh-CN") {
  document.cookie = `locale=${locale}; path=/`;
  return render(
    <I18nProvider initialLocale={locale}>
      <ScheduledOriginDivider origin={origin} />
    </I18nProvider>,
  );
}

describe("ScheduledOriginDivider", () => {
  test("separates a heartbeat turn with a labeled rule", () => {
    renderDivider(ORIGIN, "en-US");
    const divider = screen.getByTestId("scheduled-origin-divider");
    expect(divider.getAttribute("role")).toBe("separator");
    expect(divider.getAttribute("data-task-id")).toBe(
      "persistent-bot-heartbeat",
    );
    expect(divider.textContent).toContain("Scheduled run");
    expect(divider.getAttribute("aria-label")).toContain("Scheduled run");
  });

  test("labels a trial run and reads in Chinese", () => {
    renderDivider({ ...ORIGIN, trigger: "manual" }, "zh-CN");
    expect(
      screen.getByTestId("scheduled-origin-divider").textContent,
    ).toContain("试运行");
  });

  test("is driven by the message-level deerflow_scheduled_origin metadata", () => {
    const message = {
      type: "human",
      id: "scheduled-run-1",
      content: "host-written launch text",
      additional_kwargs: { deerflow_scheduled_origin: ORIGIN },
    } as Parameters<typeof scheduledOriginOf>[0];
    const origin = scheduledOriginOf(message);
    expect(origin).not.toBeNull();
    renderDivider(origin!, "en-US");
    expect(screen.getByTestId("scheduled-origin-divider")).toBeTruthy();

    const ordinary = {
      type: "human",
      id: "plain-1",
      content: "hello",
      additional_kwargs: {},
    } as Parameters<typeof scheduledOriginOf>[0];
    expect(scheduledOriginOf(ordinary)).toBeNull();
  });
});
