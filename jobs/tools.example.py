"""Example tool functions for jobs/workplace.json: each gets the call's arguments as a dict and returns a dict
(the record, or what the system did), or raises an error with a message the worker can read."""

ORDERS = {"1043": {"order_id": "1043", "total": "25.00", "status": "lost in transit"}}


def get_order(args):
    order = ORDERS.get(str(args.get("order_id", "")).strip())
    if order is None:
        raise KeyError("no order %s" % args.get("order_id"))
    return order
