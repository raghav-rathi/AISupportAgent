---
id: troubleshooting-sync-and-performance
title: Troubleshooting sync and performance
tiers: free, pro, enterprise
---

# Troubleshooting sync and performance

## Changes are not syncing

A grey cloud icon at the top right means the app is offline or cannot reach Fernhill.

1. Check status.fernhill.example for an open incident.
2. Reload the page, or quit and reopen the app.
3. Check that your network allows WebSocket connections to `*.fernhill.example` on port 443. Some company firewalls block them.
4. Sign out and sign in again.

Edits made while offline are kept on the device and sent when the connection returns. If two people changed the same card while one was offline, the newer edit wins and the older one is kept in version history.

## A board loads slowly

Boards with more than 5,000 cards load slowly. Archive finished cards or split the board.

1. Open the board menu and select **Archive cards**.
2. Choose **Done for more than 30 days**.
3. Confirm. Archived cards stay searchable.

## Supported browsers

Fernhill supports the two most recent versions of Chrome, Firefox, Safari and Edge. Older browsers can show a blank page or miss live updates.

## Clearing the app cache

1. Open **Settings > Advanced**.
2. Select **Clear local cache**.
3. Reload the page.

Clearing the cache does not remove any content. It only removes the copy stored on the device.

## Mobile notifications are not arriving

1. In the phone's settings, check that notifications are allowed for Fernhill.
2. In the app, open **Profile > Notifications** and check that push is on for the events you want.
3. Turn off battery saver for Fernhill. On Android, battery saver often delays push messages.
4. Sign out of the app and sign in again to register the device anew.
