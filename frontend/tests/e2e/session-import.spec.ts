import { expect, test } from "@playwright/test";

import {
  mockLangGraphAPI,
  THREAD_IMPORTED_METADATA_KEY,
} from "./utils/mock-api";

const IMPORTED_THREAD_ID = "00000000-0000-0000-0000-0000000000bb";

const EXPORT_PAYLOAD = {
  title: "Imported research chat",
  thread_id: "old-thread-ignored",
  created_at: "2026-08-30T10:00:00Z",
  exported_at: "2026-08-31T10:00:00Z",
  messages: [
    { type: "human", id: "h1", content: "What is the Sharpe ratio?" },
    { type: "ai", id: "a1", content: "A risk-adjusted return measure." },
  ],
};

test.describe("Session import", () => {
  test("imports an exported JSON file and navigates to the new chat", async ({
    page,
  }) => {
    mockLangGraphAPI(page);

    let receivedBody: unknown = null;
    void page.route("**/api/threads/import", (route) => {
      receivedBody = route.request().postDataJSON();
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          thread_id: IMPORTED_THREAD_ID,
          status: "idle",
          created_at: "2026-08-31T12:00:00Z",
          updated_at: "2026-08-31T12:00:00Z",
          metadata: { [THREAD_IMPORTED_METADATA_KEY]: true },
          imported_message_count: 2,
        }),
      });
    });

    await page.goto("/workspace/chats/new");

    const input = page.locator('[data-testid="import-chat-file-input"]');
    await input.setInputFiles({
      name: "export.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(EXPORT_PAYLOAD)),
    });

    await expect(page).toHaveURL(`/workspace/chats/${IMPORTED_THREAD_ID}`, {
      timeout: 15_000,
    });
    expect(receivedBody).toMatchObject({
      title: "Imported research chat",
    });
    expect((receivedBody as { messages?: unknown[] }).messages).toHaveLength(2);
  });

  test("shows an error toast when the import is rejected", async ({ page }) => {
    mockLangGraphAPI(page);

    void page.route("**/api/threads/import", (route) =>
      route.fulfill({
        status: 400,
        contentType: "application/json",
        body: JSON.stringify({
          detail: "Import contains no importable messages",
        }),
      }),
    );

    await page.goto("/workspace/chats/new");

    const input = page.locator('[data-testid="import-chat-file-input"]');
    await input.setInputFiles({
      name: "bad-export.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify({ messages: [] })),
    });

    await expect(
      page.getByText("Import failed. Check the file and try again."),
    ).toBeVisible({ timeout: 15_000 });
    await expect(page).toHaveURL("/workspace/chats/new");
  });

  test("rejects a file that is not valid JSON before calling the API", async ({
    page,
  }) => {
    mockLangGraphAPI(page);

    let importCalled = false;
    void page.route("**/api/threads/import", (route) => {
      importCalled = true;
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ thread_id: IMPORTED_THREAD_ID }),
      });
    });

    await page.goto("/workspace/chats/new");

    const input = page.locator('[data-testid="import-chat-file-input"]');
    await input.setInputFiles({
      name: "corrupt.json",
      mimeType: "application/json",
      buffer: Buffer.from("{not json"),
    });

    await expect(
      page.getByText("Import failed. Check the file and try again."),
    ).toBeVisible({ timeout: 15_000 });
    expect(importCalled).toBe(false);
  });
});
