import { beforeEach, expect, test, rs } from "@rstest/core";

const fetchWithAuth = rs.fn();

rs.mock("@/core/api/fetcher", () => ({
  fetch: fetchWithAuth,
}));

beforeEach(() => {
  fetchWithAuth.mockReset();
});

test("importThreadSession posts the export payload to the import endpoint", async () => {
  fetchWithAuth.mockResolvedValue({
    ok: true,
    json: async () => ({
      thread_id: "imported-thread-1",
      status: "idle",
      metadata: { deerflow_imported: true },
      imported_message_count: 4,
    }),
  });

  const { importThreadSession } = await import("@/core/threads/api");

  const payload = {
    title: "Research chat",
    messages: [
      { type: "human", content: "hello" },
      { type: "ai", content: "hi there" },
    ],
  };

  await expect(importThreadSession(payload)).resolves.toMatchObject({
    thread_id: "imported-thread-1",
    imported_message_count: 4,
  });

  expect(fetchWithAuth).toHaveBeenCalledWith(
    expect.stringContaining("/api/threads/import"),
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
  );
});

test("importThreadSession surfaces gateway detail on failure", async () => {
  fetchWithAuth.mockResolvedValue({
    ok: false,
    json: async () => ({
      detail: "Import contains no importable messages",
    }),
  });

  const { importThreadSession } = await import("@/core/threads/api");

  await expect(importThreadSession({ messages: [] })).rejects.toThrow(
    "Import contains no importable messages",
  );
});
