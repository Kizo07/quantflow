import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { throwGatewayApiError } from "@/core/api/errors";
import { fetch } from "@/core/api/fetcher";
import { getBackendBaseURL } from "@/core/config";

export type PersistentBotStatus = {
  enabled: boolean;
  seeded: boolean;
  user_paused: boolean;
  agent_name: string;
  thread_id: string | null;
  heartbeat_task_id: string | null;
  heartbeat_status: string | null;
};

export const PERSISTENT_BOT_STATUS_QUERY_KEY = ["persistent-bot"] as const;

function persistentBotUrl(path = ""): string {
  return `${getBackendBaseURL()}/api/persistent-bot${path}`;
}

export async function fetchPersistentBotStatus(): Promise<PersistentBotStatus> {
  const response = await fetch(persistentBotUrl(), { cache: "no-store" });
  if (!response.ok) {
    await throwGatewayApiError(
      response,
      `Failed to load assistant status: ${response.statusText}`,
    );
  }
  return response.json();
}

async function postPersistentBotAction(
  path: "/pause" | "/resume",
): Promise<PersistentBotStatus> {
  const response = await fetch(persistentBotUrl(path), { method: "POST" });
  if (!response.ok) {
    await throwGatewayApiError(
      response,
      `Failed to update assistant: ${response.statusText}`,
    );
  }
  return response.json();
}

/**
 * Bot status for the pinned-assistant section. Only enabled once a bot
 * thread exists (the section renders nothing before that), so deployments
 * without the bot never call the endpoint.
 */
export function usePersistentBotStatus(enabled: boolean) {
  return useQuery({
    queryKey: PERSISTENT_BOT_STATUS_QUERY_KEY,
    queryFn: fetchPersistentBotStatus,
    enabled,
    staleTime: 15_000,
  });
}

export function useSetPersistentBotPaused() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (paused: boolean) =>
      postPersistentBotAction(paused ? "/pause" : "/resume"),
    onSuccess: (status) => {
      queryClient.setQueryData(PERSISTENT_BOT_STATUS_QUERY_KEY, status);
    },
  });
}
