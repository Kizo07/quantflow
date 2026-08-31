"use client";

import { UsageDashboard } from "@/components/workspace/usage/usage-dashboard";
import {
  WorkspaceBody,
  WorkspaceContainer,
  WorkspaceHeader,
} from "@/components/workspace/workspace-container";
import { useI18n } from "@/core/i18n/hooks";

export default function UsagePage() {
  const { t } = useI18n();
  return (
    <WorkspaceContainer>
      <WorkspaceHeader />
      <WorkspaceBody>
        <div className="mx-auto w-full max-w-3xl px-4 py-6">
          <h1 className="mb-4 text-2xl font-semibold">{t.usage.title}</h1>
          <UsageDashboard />
        </div>
      </WorkspaceBody>
    </WorkspaceContainer>
  );
}
