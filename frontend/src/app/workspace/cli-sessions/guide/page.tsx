"use client";

import Link from "next/link";
import { useEffect } from "react";

import {
  WorkspaceBody,
  WorkspaceContainer,
  WorkspaceHeader,
} from "@/components/workspace/workspace-container";
import {
  CLI_APPROVAL_MODES,
  CLI_EFFORT_TIERS,
  CLI_MODEL_MUSE_13,
  CLI_PROVIDERS,
} from "@/core/cli-sessions/types";
import { useI18n } from "@/core/i18n/hooks";

export default function CliSessionsGuidePage() {
  const { t } = useI18n();
  const guide = t.cliSessions.guide;

  useEffect(() => {
    document.title = `${guide.title} - ${t.pages.appName}`;
  }, [guide.title, t.pages.appName]);

  const flagRows: Array<{ flag: string; values: string; note: string }> = [
    {
      flag: "--model",
      values: `${CLI_MODEL_MUSE_13} / …`,
      note: guide.noteModel,
    },
    {
      flag: "--provider",
      values: CLI_PROVIDERS.join(" / "),
      note: guide.noteProvider,
    },
    {
      flag: "--reasoning-effort",
      values: CLI_EFFORT_TIERS.join(" / "),
      note: guide.noteEffort,
    },
    {
      flag: "--contributor-mode",
      values: "on / off",
      note: guide.noteContributor,
    },
    {
      flag: "--approval-mode",
      values: CLI_APPROVAL_MODES.join(" / "),
      note: guide.noteApproval,
    },
    {
      flag: "--sandbox-network",
      values: "profile name",
      note: guide.noteSandbox,
    },
    { flag: "--trust-workspace", values: "flag", note: guide.noteTrust },
    { flag: "--yolo", values: "flag + confirm", note: guide.noteYolo },
  ];

  return (
    <WorkspaceContainer>
      <WorkspaceHeader />
      <WorkspaceBody>
        <div className="mx-auto flex w-full max-w-4xl min-w-0 flex-col gap-6 px-4 py-5 sm:px-6">
          <div className="flex min-w-0 flex-col gap-1">
            <h1 className="text-2xl font-semibold">{guide.title}</h1>
            <p className="text-muted-foreground text-sm">{guide.intro}</p>
          </div>

          <section className="flex flex-col gap-2">
            <h2 className="text-lg font-semibold">{guide.flagsTitle}</h2>
            <p className="text-muted-foreground text-sm">{guide.flagsIntro}</p>
            <div className="overflow-x-auto rounded-lg border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left">
                    <th className="px-3 py-2 font-medium">
                      {guide.flagColumn}
                    </th>
                    <th className="px-3 py-2 font-medium">
                      {guide.valuesColumn}
                    </th>
                    <th className="px-3 py-2 font-medium">
                      {guide.notesColumn}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {flagRows.map((row) => (
                    <tr key={row.flag} className="border-b last:border-0">
                      <td className="px-3 py-2 font-mono text-[13px] whitespace-nowrap">
                        {row.flag}
                      </td>
                      <td className="px-3 py-2 font-mono text-[13px]">
                        {row.values}
                      </td>
                      <td className="px-3 py-2">{row.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section
            data-testid="cli-guide-yolo-warning"
            className="border-destructive/40 bg-destructive/5 flex flex-col gap-1 rounded-lg border p-4"
          >
            <h2 className="text-lg font-semibold">{guide.yoloTitle}</h2>
            <p className="text-sm">{guide.yoloBody}</p>
          </section>

          <section className="flex flex-col gap-1">
            <h2 className="text-lg font-semibold">{guide.logsTitle}</h2>
            <p className="text-sm break-words">{guide.logsBody}</p>
          </section>

          <section className="flex flex-col gap-1">
            <h2 className="text-lg font-semibold">{guide.modeTitle}</h2>
            <p className="text-sm">{guide.modeBody}</p>
          </section>

          <section className="flex flex-col gap-1">
            <h2 className="text-lg font-semibold">{guide.presetTitle}</h2>
            <p className="text-sm">{guide.presetBody}</p>
          </section>

          <section className="flex flex-col gap-1">
            <h2 className="text-lg font-semibold">{guide.chatTitle}</h2>
            <p className="text-sm">{guide.chatBody}</p>
          </section>

          <div>
            <Link
              href="/workspace/cli-sessions"
              className="text-primary text-sm underline underline-offset-4"
            >
              {guide.backToSessions}
            </Link>
          </div>
        </div>
      </WorkspaceBody>
    </WorkspaceContainer>
  );
}
