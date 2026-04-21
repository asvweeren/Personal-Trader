import { useEffect, useRef, useState, useCallback } from "react";
import type { WSMessage } from "../types";

const TOKEN_KEY = "auth_token";
const MAX_RECONNECT_DELAY = 60000;
const INITIAL_RECONNECT_DELAY = 3000;
const MAX_AUTH_FAILURES = 3;

export function useWebSocket(url: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);
  const reconnectTimeout = useRef<ReturnType<typeof setTimeout>>(undefined);
  const reconnectDelay = useRef(INITIAL_RECONNECT_DELAY);
  const authFailures = useRef(0);

  const connect = useCallback(() => {
    try {
      const token = localStorage.getItem(TOKEN_KEY);
      if (!token) {
        return;
      }

      const wsUrl = token
        ? `${url}${url.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}`
        : url;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setConnected(true);
        reconnectDelay.current = INITIAL_RECONNECT_DELAY;
        authFailures.current = 0;
        const pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "ping" }));
          }
        }, 30000);
        ws.addEventListener("close", () => clearInterval(pingInterval));
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data) as WSMessage;
          if (msg.type !== "pong") {
            setLastMessage(msg);
          }
        } catch {
          // ignore parse errors
        }
      };

      ws.onclose = (event) => {
        setConnected(false);

        // 1008 = policy violation (auth), 403 comes as code 1006 with no prior open
        const isAuthError = event.code === 1008 || (!event.wasClean && event.code === 1006 && authFailures.current >= 0 && !ws.OPEN);

        if (isAuthError) {
          authFailures.current++;
        }

        if (authFailures.current >= MAX_AUTH_FAILURES) {
          localStorage.removeItem(TOKEN_KEY);
          window.dispatchEvent(new Event("auth:logout"));
          return;
        }

        // Exponential backoff
        reconnectTimeout.current = setTimeout(connect, reconnectDelay.current);
        reconnectDelay.current = Math.min(reconnectDelay.current * 2, MAX_RECONNECT_DELAY);
      };

      ws.onerror = () => {
        ws.close();
      };

      wsRef.current = ws;
    } catch {
      reconnectTimeout.current = setTimeout(connect, reconnectDelay.current);
      reconnectDelay.current = Math.min(reconnectDelay.current * 2, MAX_RECONNECT_DELAY);
    }
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimeout.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { connected, lastMessage };
}
