import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "@rstest/core";

const frontendRoot = join(import.meta.dirname, "../../../..");

describe("decorative animation scheduling", () => {
  it("suspends the Galaxy render loop when its container is inactive", () => {
    const source = readFileSync(
      join(frontendRoot, "src/components/landing/hero.tsx"),
      "utf8",
    );

    expect(source).toContain("useRenderActivity");
    expect(source).toContain("renderGalaxy && (");
    expect(source).toContain(
      'dynamic(() => import("@/components/ui/galaxy"), { ssr: false })',
    );
  });

  it("honors reduced motion for hero decoration and word rotation", () => {
    const source = readFileSync(
      join(frontendRoot, "src/components/landing/hero.tsx"),
      "utf8",
    );

    expect(source).toContain("usePrefersReducedMotion");
    expect(source).toContain("{!reducedMotion && (");
    expect(source).toContain("if (reducedMotion) return;");
  });

  it("does not load the skills animation before its section is visible", () => {
    const source = readFileSync(
      join(frontendRoot, "src/components/landing/sections/skills-section.tsx"),
      "utf8",
    );

    expect(source).toContain('import("../progressive-skills-animation")');
    expect(source).toContain("ssr: false");
    expect(source).toContain("useRenderActivity(animationRef, false)");
    expect(source).toContain(
      "renderAnimation && <ProgressiveSkillsAnimation />",
    );
  });
});
