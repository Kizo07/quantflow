export interface ConsoleStatsResponse {
  total_runs: number;
  active_runs: number;
  failed_runs: number;
  total_threads: number;
  total_agents: number;
  total_tokens: number;
  total_cost: number | null;
  currency: string | null;
}

export interface ConsoleRunItem {
  run_id: string;
  thread_id: string;
  thread_title: string | null;
  assistant_id: string | null;
  status: string;
  model_name: string | null;
  created_at: string | null;
  updated_at: string | null;
  duration_seconds: number | null;
  total_tokens: number;
  message_count: number;
  cost: number | null;
  error: string | null;
}

export interface ConsoleRunsResponse {
  runs: ConsoleRunItem[];
  has_more: boolean;
}

export interface ConsoleUsageDay {
  date: string;
  total_tokens: number;
  input_tokens: number;
  output_tokens: number;
  runs: number;
  cost: number;
}

export interface ConsoleUsageModelBreakdown {
  tokens: number;
  runs: number;
  cost: number | null;
  input_tokens: number;
  cache_read_tokens: number;
}

export interface ConsoleUsageResponse {
  days: ConsoleUsageDay[];
  by_model: Record<string, ConsoleUsageModelBreakdown>;
  total_tokens: number;
  total_runs: number;
  total_cost: number | null;
  currency: string | null;
}
