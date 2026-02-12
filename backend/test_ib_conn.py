import asyncio
from ib_insync import IB

async def t():
    ib = IB()
    try:
        await ib.connectAsync(host="ib-gateway", port=4002, clientId=99, timeout=15)
        sv = ib.client.serverVersion()
        print("OK server=" + str(sv))
        ib.disconnect()
    except Exception as e:
        print("FAIL:" + type(e).__name__ + ":" + str(e))

asyncio.run(t())
