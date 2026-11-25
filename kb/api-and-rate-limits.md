---
id: api-and-rate-limits
title: API and rate limits
tiers: free, pro, enterprise
---

# API and rate limits

The Fernhill REST API lives at `https://api.fernhill.example/v1`. Every plan includes API access.

## API keys

1. Open **Settings > Developer > API keys**.
2. Select **New key**, give it a name and choose read or read-write access.
3. Copy the key. It is shown once.

To rotate a key, create a new one, move your code to it, then select **Revoke** on the old key. A revoked key stops working at once. Keys belong to the workspace, not to the member who created them.

## Rate limits

### Free
<!-- tiers: free -->

A Free workspace can make 60 API requests per minute.

### Pro
<!-- tiers: pro -->

A Pro workspace can make 600 API requests per minute.

### Enterprise
<!-- tiers: enterprise -->

An Enterprise workspace can make 6,000 API requests per minute. A higher limit can be agreed with your account manager.

## When you hit the limit

The limit is counted per workspace across all keys. A request over the limit gets HTTP status 429 with a `Retry-After` header that gives the number of seconds to wait. Wait that long, then retry. Requests that got a 429 do not count toward the limit.

## Pagination

List endpoints return up to 100 items per page. The response includes a `next_cursor` value. Pass it as the `cursor` parameter to get the next page. A missing `next_cursor` means the last page.
