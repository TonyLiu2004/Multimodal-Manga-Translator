import AsyncStorage from "@react-native-async-storage/async-storage";
import type { SupabaseClient } from "@supabase/supabase-js";

const LOCAL_SESSION_STARTED_AT_KEY = "@manglify_local_session_started_at";

const MAX_SESSION_MS = 3 * 24 * 60 * 60 * 1000; //auto sign out after 3 days

/**
 * If there is no session, clears the local clock.
 * If there is a session, ensures a start timestamp exists (first time only).
 * If the session is older than MAX_SESSION_MS, signs out and clears the clock.
 */
export async function applyLocalSessionMaxAgePolicy(
  supabase: SupabaseClient
): Promise<void> {
  const {
    data: { session },
  } = await supabase.auth.getSession();

  if (!session) {
    await AsyncStorage.removeItem(LOCAL_SESSION_STARTED_AT_KEY);
    return;
  }

  const raw = await AsyncStorage.getItem(LOCAL_SESSION_STARTED_AT_KEY);
  if (!raw) {
    await AsyncStorage.setItem(LOCAL_SESSION_STARTED_AT_KEY, String(Date.now()));
    return;
  }

  const started = Number(raw);
  if (!Number.isFinite(started) || Date.now() - started > MAX_SESSION_MS) {
    await AsyncStorage.removeItem(LOCAL_SESSION_STARTED_AT_KEY);
    await supabase.auth.signOut();
  }
}
