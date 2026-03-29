import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { AppState, type AppStateStatus } from "react-native";
import type { Session } from "@supabase/supabase-js";
import { applyLocalSessionMaxAgePolicy } from "@/lib/authSessionMaxAge";
import { getSupabase, isSupabaseConfigured } from "@/lib/supabase";

export function sessionUserLabel(session: Session): string {
  const meta = session.user.user_metadata ?? {};
  const fromMeta =
    (typeof meta.display_name === "string" && meta.display_name.trim()) ||
    (typeof meta.full_name === "string" && meta.full_name.trim()) ||
    (typeof meta.name === "string" && meta.name.trim()) ||
    "";
  if (fromMeta) return fromMeta;
  return session.user.email?.trim() || "Account";
}

export type AuthContextValue = {
  session: Session | null;
  /** True until the first `getSession()` finishes when Supabase is configured. */
  loading: boolean;
  userLabel: string | null;
  signOut: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(isSupabaseConfigured());

  useEffect(() => {
    if (!isSupabaseConfigured()) {
      setSession(null);
      setLoading(false);
      return;
    }

    const supabase = getSupabase();

    const runPolicy = () => {
      void applyLocalSessionMaxAgePolicy(supabase);
    };

    supabase.auth.getSession().then(({ data: { session: s } }) => {
      setSession(s);
      setLoading(false);
      runPolicy();
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, s) => {
      setSession(s);
      runPolicy();
    });

    const onAppState = (state: AppStateStatus) => {
      if (state === "active") runPolicy();
    };
    const appSub = AppState.addEventListener("change", onAppState);

    return () => {
      subscription.unsubscribe();
      appSub.remove();
    };
  }, []);

  const signOut = useCallback(async () => {
    if (!isSupabaseConfigured()) return;
    await getSupabase().auth.signOut();
  }, []);

  const userLabel = useMemo(
    () => (session ? sessionUserLabel(session) : null),
    [session]
  );

  const value = useMemo<AuthContextValue>(
    () => ({
      session,
      loading,
      userLabel,
      signOut,
    }),
    [session, loading, userLabel, signOut]
  );

  return (
    <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return ctx;
}
