"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tooltip } from "@/components/workspace/tooltip";

/**
 * Shared composer-option primitives (D-1). Compact label + control rows for
 * composer footers and Run-options disclosures, styled to the research
 * composer's picker pills (`h-7 text-xs`, muted labels). Extracted from the
 * CLI composer so launch/send (and future composers) share one definition.
 */
export function ComposerOptionSelect({
  label,
  value,
  onChange,
  options,
  testId,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: Array<{ value: string; label: string }>;
  testId?: string;
}) {
  return (
    <span className="inline-flex min-w-0 items-center gap-1">
      <span aria-hidden="true" className="text-muted-foreground text-xs">
        {label}
      </span>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger
          aria-label={label}
          data-testid={testId}
          className="h-7 w-auto min-w-0 gap-1 px-2 text-xs"
        >
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </span>
  );
}

/** Shared compact footer toggle (D-1); see `ComposerOptionSelect`. */
export function ComposerOptionToggle({
  label,
  checked,
  onChange,
  tooltip,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  tooltip?: string;
}) {
  const control = (
    <span className="inline-flex items-center gap-1.5">
      <Switch aria-label={label} checked={checked} onCheckedChange={onChange} />
      <span aria-hidden="true" className="text-muted-foreground text-xs">
        {label}
      </span>
    </span>
  );
  return tooltip ? <Tooltip content={tooltip}>{control}</Tooltip> : control;
}
