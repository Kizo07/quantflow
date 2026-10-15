"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  useCliSession,
  useCliSessionEvents,
  useExportCliSession,
  useInspectCliSession,
  useResumeCliSession,
  useStopCliSession,
} from "@/core/cli-sessions/hooks";
import { isCliSessionActive, type CliSession } from "@/core/cli-sessions/types";
import { useI18n } from "@/core/i18n/hooks";

function formatTime(value: string | null, locale: string): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(locale, {
    dateStyle: "medium",
    timeStyle: "medium",
  }).format(date);
}

function summarizeEventPayload(payload: unknown): string {
  if (typeof payload === "string") return payload;
  if (typeof payload === "object" && payload !== null) {
    const record = payload as Record<string, unknown>;
    for (const key of ["text", "message", "content", "type", "event"]) {
      const value = record[key];
      if (typeof value === "string" && value.length > 0) return value;
    }
  }
  try {
    return JSON.stringify(payload);
  } catch {
    return String(payload);
  }
}

function downloadJson(filename: string, document: unknown) {
  const blob = new Blob(
    [
      typeof document === "string"
        ? document
        : JSON.stringify(document, null, 2),
    ],
    { type: "application/json" },
  );
  const url = URL.createObjectURL(blob);
  const anchor = window.document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function CliSessionDetail({ session }: { session: CliSession }) {
  const { t, locale } = useI18n();
  const st = t.cliSessions.detail;
  const query = useCliSession(session.id, { initialData: session });
  const current = query.data ?? session;
  const active = isCliSessionActive(current);

  const [resumePrompt, setResumePrompt] = useState("");
  const [redacted, setRedacted] = useState(true);
  const [inspectOutput, setInspectOutput] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const events = useCliSessionEvents(current.id, active);
  const resume = useResumeCliSession();
  const stop = useStopCliSession();
  const exportRun = useExportCliSession();
  const inspect = useInspectCliSession();

  const fail = (fallback: string) => (error: Error) =>
    setActionError(error.message || fallback);

  const handleResume = () => {
    setActionError(null);
    resume.mutate(
      {
        sessionId: current.id,
        payload: {
          prompt: resumePrompt.trim() || null,
        },
      },
      { onError: fail(t.cliSessions.errors.resumeFailed) },
    );
  };

  const handleStop = () => {
    setActionError(null);
    stop.mutate(current.id, {
      onError: fail(t.cliSessions.errors.stopFailed),
    });
  };

  const handleExport = () => {
    setActionError(null);
    exportRun.mutate(
      { sessionId: current.id, redacted },
      {
        onSuccess: (result) =>
          downloadJson(`cli-session-${current.id}.json`, result.document),
        onError: fail(t.cliSessions.errors.exportFailed),
      },
    );
  };

  const handleInspect = () => {
    setActionError(null);
    setInspectOutput(null);
    inspect.mutate(current.id, {
      onSuccess: (result) => setInspectOutput(result.output),
      onError: fail(t.cliSessions.errors.inspectFailed),
    });
  };

  const rows: Array<[string, string]> = [
    [st.model, current.model ?? t.cliSessions.launch.modelCliDefault],
    [st.effort, current.effort ?? t.cliSessions.launch.effortNone],
    [st.provider, current.provider ?? t.cliSessions.launch.providerNone],
    [st.started, formatTime(current.started_at, locale)],
    [st.ended, formatTime(current.ended_at, locale)],
    [st.workspace, current.workspace ?? "—"],
  ];

  return (
    <div
      className="flex min-w-0 flex-col gap-4"
      data-testid="cli-session-detail"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 flex-col gap-1">
          <h2 className="text-lg font-semibold">
            {st.header}{" "}
            <span className="font-mono text-sm font-normal">
              {current.id.slice(0, 8)}
            </span>
          </h2>
          <p className="text-muted-foreground text-sm">
            {t.cliSessions.status[current.status]}
            {current.resumed_from && (
              <>
                {" · "}
                {st.resumedFrom} {current.resumed_from.slice(0, 8)}
              </>
            )}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {active && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleStop}
              disabled={stop.isPending}
            >
              {stop.isPending ? st.stopping : st.stop}
            </Button>
          )}
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={handleInspect}
            disabled={inspect.isPending}
          >
            {inspect.isPending ? st.inspecting : st.inspect}
          </Button>
          <label className="flex items-center gap-1.5 text-sm">
            <Switch
              aria-label={st.redacted}
              checked={redacted}
              onCheckedChange={setRedacted}
            />
            {st.redacted}
          </label>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={handleExport}
            disabled={exportRun.isPending}
          >
            {exportRun.isPending ? st.exporting : st.export}
          </Button>
        </div>
      </div>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm md:grid-cols-3">
        {rows.map(([label, value]) => (
          <div key={label} className="flex min-w-0 flex-col">
            <dt className="text-muted-foreground text-xs">{label}</dt>
            <dd className="truncate">{value}</dd>
          </div>
        ))}
      </dl>

      {current.error && (
        <p role="alert" className="text-destructive text-sm">
          {st.error}: {current.error}
        </p>
      )}

      <div className="flex flex-col gap-2 rounded-lg border p-3">
        <div className="flex flex-col gap-2 sm:flex-row">
          <Input
            aria-label={st.resume}
            placeholder={st.resumePromptPlaceholder}
            value={resumePrompt}
            onChange={(event) => setResumePrompt(event.target.value)}
          />
          <Button
            type="button"
            variant="secondary"
            onClick={handleResume}
            disabled={resume.isPending}
          >
            {resume.isPending ? st.resuming : st.resume}
          </Button>
        </div>
      </div>

      {actionError && (
        <p role="alert" className="text-destructive text-sm">
          {actionError}
        </p>
      )}

      <section aria-label={st.events} className="flex flex-col gap-2">
        <h3 className="text-sm font-semibold">
          {st.events}
          <span className="text-muted-foreground ml-2 font-normal">
            {events.unavailable
              ? st.eventsUnavailable
              : active
                ? events.connected
                  ? st.eventsLive
                  : st.eventsEmpty
                : events.exitCode !== null
                  ? `${st.eventsFinished} (${events.exitCode})`
                  : st.eventsFinished}
          </span>
        </h3>
        {events.events.length > 0 && (
          <ol
            data-testid="cli-session-events"
            className="bg-muted/40 flex max-h-64 flex-col gap-1 overflow-y-auto rounded-lg p-3 font-mono text-xs"
          >
            {events.events.map((event) => (
              <li key={event.seq} className="break-words whitespace-pre-wrap">
                {summarizeEventPayload(event.payload)}
              </li>
            ))}
          </ol>
        )}
      </section>

      {inspectOutput !== null && (
        <section aria-label={st.inspect} className="flex flex-col gap-2">
          <h3 className="text-sm font-semibold">{st.inspect}</h3>
          <pre
            data-testid="cli-session-inspect"
            className="bg-muted/40 max-h-64 overflow-auto rounded-lg p-3 font-mono text-xs whitespace-pre-wrap"
          >
            {inspectOutput || st.inspectEmpty}
          </pre>
        </section>
      )}
    </div>
  );
}
