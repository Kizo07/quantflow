"use client";

import { BotIcon } from "lucide-react";

import { type Agent } from "@/core/agents";

import { WelcomeShell } from "./welcome";

export function AgentWelcome({
  className,
  agent,
  agentName,
}: {
  className?: string;
  agent: Agent | null | undefined;
  agentName: string;
}) {
  const displayName = agent?.display_name?.length
    ? agent.display_name
    : (agent?.name ?? agentName);
  const description = agent?.description;

  return (
    <WelcomeShell className={className}>
      <div className="bg-primary/10 flex h-12 w-12 items-center justify-center rounded-full">
        <BotIcon className="text-primary h-6 w-6" />
      </div>
      <div className="text-2xl font-bold">{displayName}</div>
      {description && (
        <p className="text-muted-foreground max-w-sm text-sm">{description}</p>
      )}
    </WelcomeShell>
  );
}
