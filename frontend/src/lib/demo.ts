/** The invented accounts a visitor can try the assistant with. They are public on purpose. */

export interface DemoAccount {
  number: string;
  pin: string;
  name: string;
  note: string;
}

export const ACCOUNTS: DemoAccount[] = [
  { number: "10482917", pin: "4721", name: "Maya", note: "Pro, monthly. INV-30417 can be refunded" },
  { number: "20573846", pin: "9034", name: "Daniel", note: "Pro, annual. INV-30388 is above the refund limit" },
  { number: "53920164", pin: "2458", name: "Elena", note: "Enterprise" },
  { number: "75148302", pin: "3390", name: "Jun", note: "Free" },
  { number: "64037291", pin: "8815", name: "Marcus", note: "Enterprise, suspended" },
  { number: "42851730", pin: "6672", name: "Oliver", note: "Pro, payment past due" },
];

/** What a customer would type to identify themselves. */
export function credentials(account: DemoAccount): string {
  return `My account number is ${account.number} and my PIN is ${account.pin}.`;
}
