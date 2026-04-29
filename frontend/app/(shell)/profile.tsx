import React, { useEffect, useState } from "react";
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Modal,
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
  const [editOpen, setEditOpen] = useState(false);
  const [draftDisplayName, setDraftDisplayName] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [passwordOpen, setPasswordOpen] = useState(false);
  const [draftPassword, setDraftPassword] = useState("");
  const [draftPasswordConfirm, setDraftPasswordConfirm] = useState("");
  const [passwordSaving, setPasswordSaving] = useState(false);
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [passwordSaved, setPasswordSaved] = useState(false);

  useEffect(() => {
    if (authLoading) return;
    if (!session) {
      router.replace("/sign-in");
      return;
    }
    const meta = session.user.user_metadata ?? {};
    const initial =
      typeof meta.display_name === "string" ? meta.display_name : "";
    setDisplayName(initial);
    setDraftDisplayName(initial);
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
      const name = draftDisplayName.trim();
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
      setDisplayName(name);
      setSaved(true);
      setEditOpen(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setSaving(false);
    }
  };

  const onSavePassword = async () => {
    setPasswordError(null);
    setPasswordSaved(false);
    if (!isSupabaseConfigured()) {
      setPasswordError("Supabase is not configured.");
      return;
    }
    const pw = draftPassword;
    const pw2 = draftPasswordConfirm;
    if (pw.length < 8) {
      setPasswordError("Password must be at least 8 characters.");
      return;
    }
    if (pw !== pw2) {
      setPasswordError("Passwords do not match.");
      return;
    }

    setPasswordSaving(true);
    try {
      const supabase = getSupabase();
      const { error: updateError } = await supabase.auth.updateUser({
        password: pw,
      });
      if (updateError) {
        setPasswordError(updateError.message);
        return;
      }
      setPasswordSaved(true);
      setPasswordOpen(false);
      setDraftPassword("");
      setDraftPasswordConfirm("");
    } catch (e) {
      setPasswordError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setPasswordSaving(false);
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
            <Text style={styles.emailLine}>
              {"Hello " +
                session.user.user_metadata?.display_name +
                " : " +
                session.user.email}
            </Text>

            <Pressable
              style={({ pressed }) => [
                styles.secondaryButton,
                pressed && styles.secondaryButtonPressed,
              ]}
              onPress={() => {
                setError(null);
                setSaved(false);
                setDraftDisplayName(displayName);
                setEditOpen(true);
              }}
            >
              <Text style={styles.secondaryButtonText}>
                Change display name
              </Text>
            </Pressable>

            {error ? <Text style={styles.error}>{error}</Text> : null}
            {saved ? (
              <Text style={styles.success}>Display name saved.</Text>
            ) : null}

            <Pressable
              style={({ pressed }) => [
                styles.secondaryButton,
                pressed && styles.secondaryButtonPressed,
              ]}
              onPress={() => {
                setPasswordError(null);
                setPasswordSaved(false);
                setDraftPassword("");
                setDraftPasswordConfirm("");
                setPasswordOpen(true);
              }}
            >
              <Text style={styles.secondaryButtonText}>Change password</Text>
            </Pressable>

            {passwordError ? (
              <Text style={styles.error}>{passwordError}</Text>
            ) : null}
            {passwordSaved ? (
              <Text style={styles.success}>Password updated.</Text>
            ) : null}
          </View>

          <Modal
            visible={editOpen}
            transparent
            animationType="fade"
            onRequestClose={() => (saving ? null : setEditOpen(false))}
          >
            <Pressable
              style={styles.modalOverlay}
              onPress={() => (saving ? null : setEditOpen(false))}
            >
              <Pressable style={styles.modalCard} onPress={() => {}}>
                  <View style={styles.modalHeader}>
                    <Text style={styles.modalTitle}>Change display name</Text>
                    <Pressable
                      style={({ pressed }) => [
                        styles.modalCloseButton,
                        pressed && !saving && styles.modalCloseButtonPressed,
                      ]}
                      onPress={() => setEditOpen(false)}
                      disabled={saving}
                      hitSlop={12}
                    >
                      <Text style={styles.modalCloseButtonText}>×</Text>
                    </Pressable>
                  </View>

                  <View style={styles.modalBody}>
                    <Text style={styles.modalHint}>
                      This name shows on your profile and reading lists.
                    </Text>
                    <TextInput
                      style={styles.modalInput}
                      placeholder="How you want to appear"
                      placeholderTextColor="#9ca3af"
                      value={draftDisplayName}
                      onChangeText={setDraftDisplayName}
                      autoCapitalize="words"
                      editable={!saving}
                      autoFocus
                      returnKeyType="done"
                      onSubmitEditing={() => (saving ? null : onSave())}
                    />
                  </View>

                <View style={styles.modalActions}>
                  <Pressable
                    style={({ pressed }) => [
                      styles.modalButton,
                      styles.modalButtonSecondary,
                      pressed && !saving && styles.secondaryButtonPressed,
                    ]}
                    onPress={() => setEditOpen(false)}
                    disabled={saving}
                  >
                    <Text style={styles.modalButtonSecondaryText}>Cancel</Text>
                  </Pressable>

                  <Pressable
                    style={({ pressed }) => [
                      styles.modalButton,
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
              </Pressable>
            </Pressable>
          </Modal>

          <Modal
            visible={passwordOpen}
            transparent
            animationType="fade"
            onRequestClose={() => (passwordSaving ? null : setPasswordOpen(false))}
          >
            <Pressable
              style={styles.modalOverlay}
              onPress={() => (passwordSaving ? null : setPasswordOpen(false))}
            >
              <Pressable style={styles.modalCard} onPress={() => {}}>
                <View style={styles.modalHeader}>
                  <Text style={styles.modalTitle}>Change password</Text>
                  <Pressable
                    style={({ pressed }) => [
                      styles.modalCloseButton,
                      pressed &&
                        !passwordSaving &&
                        styles.modalCloseButtonPressed,
                    ]}
                    onPress={() => setPasswordOpen(false)}
                    disabled={passwordSaving}
                    hitSlop={12}
                  >
                    <Text style={styles.modalCloseButtonText}>×</Text>
                  </Pressable>
                </View>

                <View style={styles.modalBody}>
                  <Text style={styles.modalHint}>
                    Choose a strong password (at least 8 characters).
                  </Text>
                  <View style={styles.modalInputStack}>
                    <TextInput
                      style={styles.modalInput}
                      placeholder="New password"
                      placeholderTextColor="#9ca3af"
                      value={draftPassword}
                      onChangeText={setDraftPassword}
                      editable={!passwordSaving}
                      secureTextEntry
                      autoFocus
                      autoCapitalize="none"
                      returnKeyType="next"
                    />
                    <TextInput
                      style={styles.modalInput}
                      placeholder="Confirm new password"
                      placeholderTextColor="#9ca3af"
                      value={draftPasswordConfirm}
                      onChangeText={setDraftPasswordConfirm}
                      editable={!passwordSaving}
                      secureTextEntry
                      autoCapitalize="none"
                      returnKeyType="done"
                      onSubmitEditing={() =>
                        passwordSaving ? null : onSavePassword()
                      }
                    />
                  </View>
                </View>

                <View style={styles.modalActions}>
                  <Pressable
                    style={({ pressed }) => [
                      styles.modalButton,
                      styles.modalButtonSecondary,
                      pressed && !passwordSaving && styles.secondaryButtonPressed,
                    ]}
                    onPress={() => setPasswordOpen(false)}
                    disabled={passwordSaving}
                  >
                    <Text style={styles.modalButtonSecondaryText}>Cancel</Text>
                  </Pressable>

                  <Pressable
                    style={({ pressed }) => [
                      styles.modalButton,
                      styles.button,
                      passwordSaving && styles.buttonDisabled,
                      pressed && !passwordSaving && styles.buttonPressed,
                    ]}
                    onPress={onSavePassword}
                    disabled={passwordSaving}
                  >
                    {passwordSaving ? (
                      <ActivityIndicator color="#fff" />
                    ) : (
                      <Text style={styles.buttonText}>Save</Text>
                    )}
                  </Pressable>
                </View>
              </Pressable>
            </Pressable>
          </Modal>
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
  displayNameValue: {
    fontSize: 18,
    fontWeight: "700",
    color: "#111827",
    textAlign: "center",
    marginBottom: 14,
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
  secondaryButton: {
    alignItems: "center",
    justifyContent: "center",
    height: 46,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#d1d5db",
    backgroundColor: "#fff",
    paddingHorizontal: 14,
    marginBottom: 10,
  },
  secondaryButtonPressed: {
    opacity: 0.85,
  },
  secondaryButtonText: {
    color: "#111827",
    fontWeight: "700",
    fontSize: 15,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(17, 24, 39, 0.45)",
    paddingHorizontal: 24,
    justifyContent: "center",
  },
  modalCard: {
    backgroundColor: "#fff",
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    ...cardShadow,
    width: "100%",
    maxWidth: 380,
    alignSelf: "center",
  },
  modalHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 10,
  },
  modalTitle: {
    fontSize: 16,
    fontWeight: "800",
    color: "#111827",
    textAlign: "center",
    flex: 1,
  },
  modalCloseButton: {
    width: 34,
    height: 34,
    borderRadius: 17,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#f3f4f6",
    borderWidth: 1,
    borderColor: "#e5e7eb",
  },
  modalCloseButtonPressed: {
    opacity: 0.85,
  },
  modalCloseButtonText: {
    fontSize: 22,
    lineHeight: 22,
    fontWeight: "700",
    color: "#111827",
    marginTop: -1,
  },
  modalBody: {
    marginBottom: 12,
  },
  modalInputStack: {
    gap: 10,
  },
  modalHint: {
    fontSize: 13,
    color: "#6b7280",
    textAlign: "center",
    marginBottom: 10,
    lineHeight: 18,
  },
  modalInput: {
    height: 50,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    borderRadius: 12,
    paddingHorizontal: 16,
    fontSize: 16,
    color: "#111827",
    backgroundColor: "#f9fafb",
  },
  modalActions: {
    flexDirection: "row",
    gap: 10,
    justifyContent: "flex-end",
    marginTop: 4,
  },
  modalButton: {
    flex: 1,
    height: 44,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
  },
  modalButtonSecondary: {
    borderWidth: 1,
    borderColor: "#d1d5db",
    backgroundColor: "#fff",
  },
  modalButtonSecondaryText: {
    color: "#111827",
    fontWeight: "700",
    fontSize: 15,
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
