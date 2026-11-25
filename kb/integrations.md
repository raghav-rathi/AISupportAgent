---
id: integrations
title: Integrations
tiers: free, pro, enterprise
---

# Integrations

Integrations are managed in **Settings > Integrations**. Only owners and admins can connect or remove them.

## Slack

Slack is available on every plan. It posts updates from a project to a Slack channel.

1. Open **Settings > Integrations > Slack** and select **Connect**.
2. Approve Fernhill in the Slack window that opens.
3. Pick a project and the channel that should receive its updates.

### Slack notifications are not arriving

1. In Slack, check that the Fernhill app is still a member of the channel. Private channels need `/invite @Fernhill`.
2. In Fernhill, open the project and check **Notifications > Slack** for the events you expect.
3. Disconnect and reconnect Slack in **Settings > Integrations**. This refreshes the permission token.

## GitHub
<!-- tiers: pro, enterprise -->

The GitHub integration links pull requests and commits to cards. Mention a card ID such as `FH-482` in a pull request title or a commit message, and the card shows the link and its status.

If links stop appearing, open **Settings > Integrations > GitHub** and confirm that the repository is in the list of granted repositories. Repositories added to GitHub after the first setup must be granted by hand.

## Google Drive
<!-- tiers: pro, enterprise -->

Attach Drive files to cards and documents. Fernhill shows a preview and keeps the sharing settings from Drive. A member without access in Drive sees a locked placeholder.

## Zapier
<!-- tiers: pro, enterprise -->

Fernhill has a Zapier app with triggers for new cards, moved cards and new comments, and actions to create and update cards. Zapier calls count toward your API rate limit.

## Webhooks
<!-- tiers: pro, enterprise -->

Webhooks send a POST request to your URL when something changes in a project.

- Each request carries an `X-Fernhill-Signature` header: an HMAC-SHA256 of the body, keyed with the webhook secret.
- A delivery that fails is retried up to 5 times over 24 hours, with a longer wait each time.
- A webhook that fails every delivery for 3 days in a row is turned off, and the owners get an email.

To turn a webhook back on, fix the endpoint, then select **Enable** in **Settings > Integrations > Webhooks**.

## Custom integrations
<!-- tiers: enterprise -->

Enterprise customers can have an integration built by the Fernhill solutions team. This is scoped and priced through your account manager.
