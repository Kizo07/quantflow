"use client";

import { useSearchParams } from "next/navigation";

import { NotebookFrame } from "@/components/workspace/notebooks/notebook-frame";
import {
  WorkspaceBody,
  WorkspaceContainer,
  WorkspaceHeader,
} from "@/components/workspace/workspace-container";

/**
 * Notebooks workspace page (Phase 3 connector).
 *
 * Hosts the embedded ReportForge notebook UI. Deep-link with
 * `?notebook=<project>/<name>.ipynb`; without it the frame opens the
 * sidecar's own picker. The frame fills the workspace body — no page
 * chrome beyond the standard header, so cells get full width.
 */
export default function NotebooksPage() {
  const searchParams = useSearchParams();
  const notebookId = searchParams.get("notebook") ?? undefined;
  return (
    <WorkspaceContainer>
      <WorkspaceHeader />
      <WorkspaceBody>
        <NotebookFrame notebookId={notebookId} className="h-full w-full" />
      </WorkspaceBody>
    </WorkspaceContainer>
  );
}
