"use client";

import { FileUp } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useRef, useState } from "react";
import { toast } from "sonner";

import { SidebarMenuButton, SidebarMenuItem } from "@/components/ui/sidebar";
import { useI18n } from "@/core/i18n/hooks";
import { importThreadSession } from "@/core/threads/api";

const MAX_IMPORT_FILE_BYTES = 10 * 1024 * 1024;

/**
 * Sidebar control that imports a session exported via the chat menu's
 * "Export as JSON". The file is parsed locally and posted to
 * ``POST /api/threads/import``; the gateway rebuilds it as a new
 * viewable thread and the UI navigates straight to it.
 */
export function ImportChatButton() {
  const { t } = useI18n();
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);

  const handleFile = useCallback(
    async (file: File) => {
      if (file.size > MAX_IMPORT_FILE_BYTES) {
        toast.error(t.sidebar.importChatError);
        return;
      }
      let payload: unknown;
      try {
        payload = JSON.parse(await file.text());
      } catch {
        toast.error(t.sidebar.importChatError);
        return;
      }
      if (
        typeof payload !== "object" ||
        payload === null ||
        !Array.isArray((payload as { messages?: unknown }).messages)
      ) {
        toast.error(t.sidebar.importChatError);
        return;
      }
      setBusy(true);
      try {
        const result = await importThreadSession(
          payload as Parameters<typeof importThreadSession>[0],
        );
        toast.success(t.sidebar.importChatSuccess);
        router.push(`/workspace/chats/${encodeURIComponent(result.thread_id)}`);
      } catch {
        toast.error(t.sidebar.importChatError);
      } finally {
        setBusy(false);
        // Reset so the same file can be re-imported after a failure.
        if (inputRef.current) inputRef.current.value = "";
      }
    },
    [router, t.sidebar.importChatError, t.sidebar.importChatSuccess],
  );

  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        data-testid="import-chat-button"
        disabled={busy}
        onClick={() => inputRef.current?.click()}
      >
        <FileUp size={16} />
        <span>{t.sidebar.importChat}</span>
      </SidebarMenuButton>
      <input
        ref={inputRef}
        type="file"
        accept=".json,application/json"
        data-testid="import-chat-file-input"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) void handleFile(file);
        }}
      />
    </SidebarMenuItem>
  );
}
