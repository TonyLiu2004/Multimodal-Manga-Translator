import React, { useRef, useState } from "react";
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
} from "react-native";
import { type Href, useRouter } from "expo-router";
import { SafeAreaView } from "react-native-safe-area-context";
import { getSupabase } from "@/lib/supabase";

function resetRedirectUrl(): string | undefined {
  if (Platform.OS === "web" && typeof window !== "undefined") {
    return `${window.location.origin}/reset-password`;
  }
  return undefined;
}

export default function ForgotPasswordScreen() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const submittingRef = useRef(false);

  const onSubmit = async () => {
    if (submittingRef.current) return;

    setMessage(null);
    setError(null);

    const trimmed = email.trim();
    if (!trimmed) {
      setError("Enter your account email.");
      return;
    }

    submittingRef.current = true;
    setLoading(true);
    try {
      const redirectTo = resetRedirectUrl();
      const { error: resetError } = await getSupabase().auth.resetPasswordForEmail(
        trimmed,
        redirectTo ? { redirectTo } : undefined,
      );

      if (resetError) {
        setError(resetError.message);
        return;
      }

      setMessage("Check your email for a password reset link.");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      submittingRef.current = false;
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safe}>
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : undefined}
        style={styles.flex}
      >
        <ScrollView
          keyboardShouldPersistTaps="handled"
          contentContainerStyle={styles.scroll}
        >
          <Pressable onPress={() => router.push("/" as Href)}>
            <Text style={styles.textCenter}>Manglify</Text>
          </Pressable>
          <Text style={styles.title}>Reset password</Text>
          <Text style={styles.hint}>
            Enter your email and we{"'"}ll send you a link to choose a new
            password.
          </Text>

          <TextInput
            style={styles.input}
            placeholder="Email"
            placeholderTextColor="#999"
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="email-address"
            value={email}
            onChangeText={setEmail}
          />

          {error ? <Text style={styles.error}>{error}</Text> : null}
          {message ? <Text style={styles.success}>{message}</Text> : null}

          <Pressable
            style={[styles.button, loading && styles.buttonDisabled]}
            onPress={onSubmit}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.buttonText}>Send reset link</Text>
            )}
          </Pressable>

          <Pressable
            onPress={() => router.push("/sign-in")}
            style={styles.linkWrap}
          >
            <Text style={styles.link}>Back to sign in</Text>
          </Pressable>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#fff" },
  flex: { flex: 1 },
  scroll: {
    padding: 24,
    alignItems: "stretch",
    maxWidth: 420,
    width: "100%",
    alignSelf: "center",
  },
  textCenter: {
    fontSize: 32,
    fontWeight: "bold",
    color: "#111",
    textAlign: "center",
    marginBottom: 8,
  },
  title: {
    fontSize: 28,
    marginBottom: 12,
    color: "#111",
  },
  hint: {
    fontSize: 14,
    color: "#666",
    marginBottom: 16,
    lineHeight: 20,
  },
  input: {
    height: 50,
    borderWidth: 1,
    borderColor: "#E0E0E0",
    borderRadius: 10,
    paddingHorizontal: 15,
    fontSize: 16,
    color: "#333",
    backgroundColor: "#FAFAFA",
    marginBottom: 12,
  },
  error: { color: "#c62828", marginBottom: 8, fontSize: 14 },
  success: { color: "#2e7d32", marginBottom: 8, fontSize: 14 },
  button: {
    backgroundColor: "#111",
    height: 50,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 8,
  },
  buttonDisabled: { opacity: 0.6 },
  buttonText: { color: "#fff", fontSize: 16, fontWeight: "600" },
  linkWrap: { marginTop: 20, alignItems: "center" },
  link: { fontSize: 15, color: "#1565c0" },
});
