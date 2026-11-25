---
id: sso-and-provisioning
title: Single sign-on and provisioning
tiers: pro, enterprise
---

# Single sign-on and provisioning

## Google sign-in

Members can sign in with a Google account in place of a password.

1. Open **Settings > Security > Sign-in methods**.
2. Turn on **Google sign-in**.
3. Optional: enter your company domain to accept only Google accounts from that domain.

## SAML single sign-on
<!-- tiers: enterprise -->

SAML single sign-on lets members sign in through your identity provider. Fernhill supports Okta, Microsoft Entra ID, Google Workspace and OneLogin.

1. Open **Settings > Security > SAML**.
2. Copy the ACS URL and the Entity ID into a new SAML app in your identity provider.
3. Paste the identity provider's metadata URL into Fernhill.
4. Select **Test sign-in** and complete one sign-in.
5. Turn on **Require SSO** once the test passes. Owners keep password sign-in as a fallback.

### SAML troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `invalid_audience` | The Entity ID in the identity provider does not match | Copy the Entity ID again from Settings > Security > SAML |
| `email_mismatch` | The NameID is not the member's email address | Set the NameID format to email address |
| `clock_skew` | The identity provider's clock is more than 5 minutes off | Sync the identity provider's clock |

## SCIM provisioning
<!-- tiers: enterprise -->

SCIM creates and removes members automatically from your identity provider. Create a SCIM token in **Settings > Security > SCIM** and paste it into the identity provider. A member removed in the identity provider is deactivated in Fernhill within 15 minutes, and their seat is freed.
