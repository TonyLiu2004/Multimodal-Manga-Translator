import React, { useEffect, useState } from "react";
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { useRouter } from "expo-router";
import { SafeAreaView } from "react-native-safe-area-context";
import { useAuth } from "@/context/AuthContext";
import { patchAppUserDisplayName } from "@/lib/readingListApi";
import { getSupabase, isSupabaseConfigured } from "@/lib/supabase";

export default function ProfileScreen() {
  const router = useRouter();
  const { session, loading: authLoading } = useAuth();
  const [displayName, setDisplayName] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (authLoading) return;
    if (!session) {
      router.replace("/sign-in");
      return;
    }
    const meta = session.user.user_metadata ?? {};
    setDisplayName(
      typeof meta.display_name === "string" ? meta.display_name : "",
    );
  }, [session, authLoading, router]);

  const onSave = async () => {
    setError(null);
    setSaved(false);
    if (!isSupabaseConfigured()) {
      setError("Supabase is not configured.");
      return;
    }
    setSaving(true);
    try {
      const supabase = getSupabase();
      const name = displayName.trim();
      const { error: updateError } = await supabase.auth.updateUser({
        data: { display_name: name },
      });
      if (updateError) {
        setError(updateError.message);
        return;
      }
      const { data: refreshed, error: refreshError } =
        await supabase.auth.refreshSession();
      if (refreshError) {
        setError(refreshError.message);
        return;
      }
      const token = refreshed.session?.access_token;
      if (!token) {
        setError("Could not refresh session after save.");
        return;
      }
      await patchAppUserDisplayName(token, name);
      setSaved(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setSaving(false);
    }
  };

  if (authLoading || !session) {
    return (
      <SafeAreaView style={styles.safe}>
        <ActivityIndicator style={styles.loader} />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safe} edges={["top", "right", "bottom"]}>
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : undefined}
        style={styles.flex}
      >
        <ScrollView
          keyboardShouldPersistTaps="handled"
          contentContainerStyle={styles.scroll}
        >
          <View style={styles.contentColumn}>
            <Text style={styles.title}>Profile</Text>
            <Text style={styles.emailLine}>{session.user.email}</Text>

            <View style={styles.profileCard}>
              <Text style={styles.fieldLabel}>Display name</Text>
              <TextInput
                style={styles.input}
                placeholder="How you want to appear"
                placeholderTextColor="#9ca3af"
                value={displayName}
                onChangeText={setDisplayName}
                autoCapitalize="words"
              />

              {error ? <Text style={styles.error}>{error}</Text> : null}
              {saved ? (
                <Text style={styles.success}>Display name saved.</Text>
              ) : null}

              <Pressable
                style={({ pressed }) => [
                  styles.button,
                  saving && styles.buttonDisabled,
                  pressed && !saving && styles.buttonPressed,
                ]}
                onPress={onSave}
                disabled={saving}
              >
                {saving ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <Text style={styles.buttonText}>Save</Text>
                )}
              </Pressable>
            </View>

          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const cardShadow = Platform.select({
  ios: {
    shadowColor: "#0f172a",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.07,
    shadowRadius: 14,
  },
  android: { elevation: 3 },
  default: {},
});

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#f3f4f6" },
  flex: { flex: 1 },
  loader: { marginTop: 40 },
  scroll: {
    paddingVertical: 24,
    paddingBottom: 40,
    alignItems: "center",
  },
  contentColumn: {
    width: "100%",
    maxWidth: 440,
    alignItems: "center",
    paddingHorizontal: 24,
  },
  title: {
    fontSize: 30,
    fontWeight: "800",
    marginBottom: 6,
    color: "#111827",
    letterSpacing: -0.5,
    textAlign: "center",
    alignSelf: "stretch",
  },
  emailLine: {
    fontSize: 14,
    color: "#6b7280",
    marginBottom: 22,
    textAlign: "center",
    alignSelf: "stretch",
  },
  profileCard: {
    backgroundColor: "#fff",
    borderRadius: 16,
    padding: 18,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    ...cardShadow,
    alignSelf: "stretch",
    width: "100%",
  },
  fieldLabel: {
    fontSize: 13,
    color: "#4b5563",
    marginBottom: 8,
    fontWeight: "600",
    textTransform: "uppercase",
    letterSpacing: 0.4,
    textAlign: "center",
  },
  input: {
    height: 50,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    borderRadius: 12,
    paddingHorizontal: 16,
    fontSize: 16,
    color: "#111827",
    backgroundColor: "#f9fafb",
    marginBottom: 12,
  },
  error: {
    color: "#b91c1c",
    marginBottom: 8,
    fontSize: 14,
    textAlign: "center",
    alignSelf: "stretch",
  },
  success: {
    color: "#15803d",
    marginBottom: 8,
    fontSize: 14,
    textAlign: "center",
    alignSelf: "stretch",
  },
  button: {
    backgroundColor: "#111827",
    height: 50,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 4,
    alignSelf: "stretch",
  },
  buttonPressed: { opacity: 0.88 },
  buttonDisabled: { opacity: 0.55 },
  buttonText: { color: "#fff", fontSize: 16, fontWeight: "600" },
  footerHint: {
    marginTop: 24,
    fontSize: 14,
    color: "#6b7280",
    textAlign: "center",
    lineHeight: 20,
    alignSelf: "stretch",
  },
});
