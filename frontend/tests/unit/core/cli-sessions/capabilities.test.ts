import { describe, expect, it } from "@rstest/core";

import {
  CLI_DEFAULT_EFFORT,
  CLI_EFFORT_TIERS,
  coerceCliEffort,
  selectCliEffortOptions,
  selectCliEffortTiers,
  type CliCapabilities,
} from "@/core/cli-sessions/types";

const CAPABILITIES: CliCapabilities = {
  models: [
    {
      id: "muse-spark-1.3",
      label: "Muse Spark 1.3",
      effortTiers: ["low", "high"],
      defaultEffort: "high",
    },
    {
      id: "muse-wonder-1.0",
      label: "Muse Wonder 1.0",
      effortTiers: ["minimal", "low"],
      defaultEffort: null,
    },
  ],
  defaultModel: "muse-spark-1.3",
  providers: [{ id: "meta", label: "Meta" }],
  defaultProvider: "meta",
};

describe("selectCliEffortTiers", () => {
  it("returns the selected model's tiers", () => {
    expect(selectCliEffortTiers(CAPABILITIES, "muse-spark-1.3")).toEqual({
      tiers: ["low", "high"],
      defaultEffort: "high",
    });
  });

  it("resolves a null model to the default model", () => {
    expect(selectCliEffortTiers(CAPABILITIES, null)).toEqual({
      tiers: ["low", "high"],
      defaultEffort: "high",
    });
  });

  it("falls back to the global tiers for unknown models", () => {
    expect(selectCliEffortTiers(CAPABILITIES, "custom-xyz")).toEqual({
      tiers: CLI_EFFORT_TIERS,
      defaultEffort: CLI_DEFAULT_EFFORT,
    });
  });

  it("falls back to the global tiers without capabilities", () => {
    expect(selectCliEffortTiers(null, "muse-spark-1.3")).toEqual({
      tiers: CLI_EFFORT_TIERS,
      defaultEffort: CLI_DEFAULT_EFFORT,
    });
    expect(selectCliEffortTiers(undefined, null)).toEqual({
      tiers: CLI_EFFORT_TIERS,
      defaultEffort: CLI_DEFAULT_EFFORT,
    });
  });

  it("falls back to the CLI default effort when the model has none", () => {
    expect(selectCliEffortTiers(CAPABILITIES, "muse-wonder-1.0")).toEqual({
      tiers: ["minimal", "low"],
      defaultEffort: CLI_DEFAULT_EFFORT,
    });
  });
});

describe("coerceCliEffort", () => {
  it("keeps a tier the model supports", () => {
    expect(coerceCliEffort(CAPABILITIES, "muse-spark-1.3", "low")).toBe("low");
  });

  it("snaps an unsupported tier to the model default", () => {
    expect(coerceCliEffort(CAPABILITIES, "muse-spark-1.3", "ultra")).toBe(
      "high",
    );
  });

  it("snaps to the first tier when the default is also unsupported", () => {
    expect(coerceCliEffort(CAPABILITIES, "muse-wonder-1.0", "ultra")).toBe(
      "minimal",
    );
  });

  it("keeps global tiers without capabilities", () => {
    expect(coerceCliEffort(null, "muse-spark-1.3", "ultra")).toBe("ultra");
  });
});

describe("selectCliEffortOptions", () => {
  it("appends the current value when it is valid but unadvertised", () => {
    expect(
      selectCliEffortOptions(CAPABILITIES, "muse-spark-1.3", "ultra"),
    ).toEqual(["low", "high", "ultra"]);
  });

  it("does not duplicate an advertised tier", () => {
    expect(
      selectCliEffortOptions(CAPABILITIES, "muse-spark-1.3", "low"),
    ).toEqual(["low", "high"]);
  });

  it("ignores unknown values and select sentinels", () => {
    expect(
      selectCliEffortOptions(CAPABILITIES, "muse-spark-1.3", "bogus"),
    ).toEqual(["low", "high"]);
    expect(
      selectCliEffortOptions(CAPABILITIES, "muse-spark-1.3", "__none__"),
    ).toEqual(["low", "high"]);
  });
});
