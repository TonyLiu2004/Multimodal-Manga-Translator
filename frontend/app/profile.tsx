import React, { useCallback, useEffect, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { useRouter } from "expo-router";
import { SafeAreaView } from "react-native-safe-area-context";
import { useAuth } from "@/context/AuthContext";
import { getSupabase, isSupabaseConfigured } from "@/lib/supabase";
import {
  createReadingList,
  deleteReadingList,
  fetchReadingLists,
  renameReadingList,
  type ReadingListCollection,
} from "@/lib/readingListApi";

export default function ProfileScreen() {
  const router = useRouter();
  const { session, loading: authLoading } = useAuth();
  const [displayName, setDisplayName] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  const [collections, setCollections] = useState<ReadingListCollection[]>([]);
  const [listLoading, setListLoading] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [newListName, setNewListName] = useState("");
  const [creatingList, setCreatingList] = useState(false);
  const [editingListId, setEditingListId] = useState<number | null>(null);
  const [editNameDraft, setEditNameDraft] = useState("");
  const [savingRename, setSavingRename] = useState(false);

  const loadReadingData = useCallback(async () => {
    if (!session?.access_token) return;
    setListError(null);
    setListLoading(true);
    try {
      const cols = await fetchReadingLists(session.access_token);
      setCollections(cols);
    } catch (e) {
      setListError(
        e instanceof Error ? e.message : "Could not load reading lists.",
      );
      setCollections([]);
    } finally {
      setListLoading(false);
    }
  }, [session?.access_token]);

  useEffect(() => {
    if (authLoading || !session?.access_token) return;
    void loadReadingData();
  }, [authLoading, session?.access_token, loadReadingData]);

  const onRefreshList = useCallback(async () => {
    if (!session?.access_token) return;
    setRefreshing(true);
    try {
      await loadReadingData();
    } finally {
      setRefreshing(false);
    }
  }, [session?.access_token, loadReadingData]);

  const onCreateList = async () => {
    if (!session?.access_token) return;
    const name = newListName.trim();
    if (!name) return;
    setCreatingList(true);
    setListError(null);
    try {
      const col = await createReadingList(session.access_token, name);
      setNewListName("");
      setCollections((prev) => [col, ...prev]);
    } catch (e) {
      setListError(
        e instanceof Error ? e.message : "Could not create reading list.",
      );
    } finally {
      setCreatingList(false);
    }
  };

  const onSaveRename = async (listId: number) => {
    if (!session?.access_token) return;
    const name = editNameDraft.trim();
    if (!name) return;
    setSavingRename(true);
    setListError(null);
    try {
      const updated = await renameReadingList(
        session.access_token,
        listId,
        name,
      );
      setCollections((prev) =>
        prev.map((c) => (c.id === listId ? updated : c)),
      );
      setEditingListId(null);
    } catch (e) {
      setListError(
        e instanceof Error ? e.message : "Could not rename reading list.",
      );
    } finally {
      setSavingRename(false);
    }
  };

  const confirmDeleteList = (c: ReadingListCollection) => {
    Alert.alert("Delete list", `Remove “${c.name}” and all manga in it?`, [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete",
        style: "destructive",
        onPress: () => void deleteListConfirmed(c.id),
      },
    ]);
  };

  const deleteListConfirmed = async (listId: number) => {
    if (!session?.access_token) return;
    setListError(null);
    try {
      await deleteReadingList(session.access_token, listId);
      setCollections((prev) => prev.filter((c) => c.id !== listId));
      if (editingListId === listId) setEditingListId(null);
    } catch (e) {
      setListError(
        e instanceof Error ? e.message : "Could not delete reading list.",
      );
    }
  };

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
      const { error: updateError } = await getSupabase().auth.updateUser({
        data: { display_name: displayName.trim() },
      });
      if (updateError) {
        setError(updateError.message);
        return;
      }
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
    <SafeAreaView style={styles.safe}>
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : undefined}
        style={styles.flex}
      >
        <ScrollView
          keyboardShouldPersistTaps="handled"
          contentContainerStyle={styles.scroll}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefreshList}
              tintColor="#374151"
              colors={["#111827"]}
            />
          }
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

          <View style={styles.listsSectionHeader}>
            <Text style={styles.sectionTitle}>Reading lists</Text>
            <Text style={styles.sectionHint}>
              Open a list to view titles and manage items. Add manga from
              search or the home carousel.
            </Text>
          </View>

          <View style={styles.newListCard}>
            <View style={styles.newListRow}>
              <TextInput
                style={[styles.input, styles.newListInput]}
                placeholder="New list name"
                placeholderTextColor="#9ca3af"
                value={newListName}
                onChangeText={setNewListName}
                returnKeyType="done"
                onSubmitEditing={() => {
                  if (newListName.trim() && !creatingList) void onCreateList();
                }}
                blurOnSubmit={false}
              />
              <Pressable
                style={({ pressed }) => [
                  styles.secondaryBtn,
                  (creatingList || !newListName.trim()) &&
                    styles.buttonDisabled,
                  pressed &&
                    !creatingList &&
                    !!newListName.trim() &&
                    styles.secondaryBtnPressed,
                ]}
                onPress={() => void onCreateList()}
                disabled={creatingList || !newListName.trim()}
              >
                {creatingList ? (
                  <ActivityIndicator size="small" color="#111827" />
                ) : (
                  <Text style={styles.secondaryBtnText}>Create</Text>
                )}
              </Pressable>
            </View>
          </View>

          {listLoading && collections.length === 0 ? (
            <ActivityIndicator style={styles.listSpinner} color="#374151" />
          ) : null}
          {listError ? (
            <View style={styles.inlineNoticeError}>
              <Text style={styles.listError}>{listError}</Text>
            </View>
          ) : null}
          {!listLoading && !listError && collections.length === 0 ? (
            <View style={styles.emptyStateCard}>
              <Text style={styles.emptyList}>
                No lists yet. Create one above.
              </Text>
            </View>
          ) : null}

          {collections.map((c) => {
            const isEditing = editingListId === c.id;
            return (
              <View key={c.id} style={styles.collectionCard}>
                <View style={styles.listCardHeader}>
                  {isEditing ? (
                    <View style={styles.renameRow}>
                      <TextInput
                        style={[styles.input, styles.renameInput]}
                        value={editNameDraft}
                        onChangeText={setEditNameDraft}
                        autoFocus
                        placeholderTextColor="#9ca3af"
                      />
                      <View style={styles.renameActions}>
                        <Pressable
                          style={({ pressed }) => [
                            styles.smallBtn,
                            pressed && styles.smallBtnPressed,
                          ]}
                          onPress={() => void onSaveRename(c.id)}
                          disabled={savingRename || !editNameDraft.trim()}
                        >
                          <Text style={styles.smallBtnText}>Save</Text>
                        </Pressable>
                        <Pressable
                          style={({ pressed }) => [
                            styles.smallBtnGhost,
                            pressed && styles.smallBtnGhostPressed,
                          ]}
                          onPress={() => setEditingListId(null)}
                        >
                          <Text style={styles.smallBtnGhostText}>Cancel</Text>
                        </Pressable>
                      </View>
                    </View>
                  ) : (
                    <Pressable
                      style={({ pressed }) => [
                        styles.listOpenPressable,
                        pressed && styles.listOpenPressed,
                      ]}
                      onPress={() =>
                        router.push({
                          pathname: "/reading-list/[id]",
                          params: { id: String(c.id), title: c.name },
                        })
                      }
                      accessibilityRole="button"
                      accessibilityLabel={`Open reading list ${c.name}`}
                    >
                      <View style={styles.listHeaderRow}>
                        <View style={styles.listHeaderTitles}>
                          <Text style={styles.listCardTitle} numberOfLines={2}>
                            {c.name}
                          </Text>
                          <Text style={styles.listCardCount}>
                            {c.manga_count} title
                            {c.manga_count === 1 ? "" : "s"}
                          </Text>
                        </View>
                        <View style={styles.countPill}>
                          <Text style={styles.countPillText}>
                            {c.manga_count}
                          </Text>
                        </View>
                        <Text style={styles.openArrow}>→</Text>
                      </View>
                    </Pressable>
                  )}
                </View>
                {!isEditing ? (
                  <View style={styles.listCardActions}>
                    <Pressable
                      style={({ pressed }) => pressed && styles.textLinkPressed}
                      onPress={() => {
                        setEditingListId(c.id);
                        setEditNameDraft(c.name);
                      }}
                    >
                      <Text style={styles.linkAction}>Rename</Text>
                    </Pressable>
                    <Text style={styles.actionSep}>·</Text>
                    <Pressable
                      style={({ pressed }) => pressed && styles.textLinkPressed}
                      onPress={() => confirmDeleteList(c)}
                    >
                      <Text style={styles.dangerAction}>Delete</Text>
                    </Pressable>
                  </View>
                ) : null}
              </View>
            );
          })}

          <Pressable
            onPress={() => router.back()}
            style={({ pressed }) => [
              styles.backLink,
              pressed && styles.backLinkPressed,
            ]}
          >
            <Text style={styles.backLinkText}>← Back</Text>
          </Pressable>
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
  listsSectionHeader: {
    marginTop: 26,
    marginBottom: 4,
    alignSelf: "stretch",
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "800",
    color: "#111827",
    marginBottom: 6,
    letterSpacing: -0.3,
    textAlign: "center",
  },
  sectionHint: {
    fontSize: 13,
    color: "#6b7280",
    lineHeight: 19,
    textAlign: "center",
  },
  newListCard: {
    backgroundColor: "#fff",
    borderRadius: 16,
    padding: 14,
    marginTop: 14,
    marginBottom: 6,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    ...cardShadow,
    alignSelf: "stretch",
    width: "100%",
  },
  newListRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  newListInput: { flex: 1, marginBottom: 0 },
  secondaryBtn: {
    height: 50,
    paddingHorizontal: 18,
    borderRadius: 12,
    borderWidth: 1.5,
    borderColor: "#111827",
    alignItems: "center",
    justifyContent: "center",
    minWidth: 92,
    backgroundColor: "#fff",
  },
  secondaryBtnPressed: { backgroundColor: "#f3f4f6" },
  secondaryBtnText: {
    fontSize: 15,
    fontWeight: "600",
    color: "#111827",
  },
  listSpinner: { marginVertical: 20 },
  listItemsSpinner: { marginVertical: 16, alignSelf: "center" },
  inlineNoticeError: {
    backgroundColor: "#fef2f2",
    borderRadius: 12,
    padding: 12,
    marginTop: 8,
    borderWidth: 1,
    borderColor: "#fecaca",
  },
  listError: { color: "#b91c1c", fontSize: 14 },
  emptyStateCard: {
    backgroundColor: "#fff",
    borderRadius: 14,
    padding: 20,
    marginTop: 12,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#e5e7eb",
    borderStyle: "dashed",
    alignSelf: "stretch",
    width: "100%",
  },
  emptyList: { fontSize: 14, color: "#6b7280", textAlign: "center" },
  collectionCard: {
    backgroundColor: "#fff",
    borderRadius: 16,
    padding: 14,
    marginTop: 12,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    ...cardShadow,
    alignSelf: "stretch",
    width: "100%",
  },
  listCardHeader: { marginBottom: 4 },
  listOpenPressable: {
    borderRadius: 12,
    paddingVertical: 10,
    paddingHorizontal: 10,
    marginHorizontal: -10,
    backgroundColor: "#f9fafb",
  },
  listOpenPressed: { opacity: 0.88 },
  listHeaderRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  openArrow: {
    fontSize: 18,
    color: "#2563eb",
    fontWeight: "600",
  },
  listHeaderTitles: { flex: 1, minWidth: 0 },
  listCardTitle: {
    fontSize: 17,
    fontWeight: "700",
    color: "#111827",
    textAlign: "left",
  },
  listCardCount: {
    fontSize: 13,
    color: "#6b7280",
    marginTop: 3,
    textAlign: "left",
  },
  countPill: {
    minWidth: 28,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 999,
    backgroundColor: "#111827",
    alignItems: "center",
    justifyContent: "center",
  },
  countPillText: {
    fontSize: 13,
    fontWeight: "700",
    color: "#fff",
  },
  listCardActions: {
    flexDirection: "row",
    alignItems: "center",
    paddingLeft: 4,
    marginTop: 4,
    marginBottom: 4,
    gap: 6,
  },
  linkAction: {
    fontSize: 14,
    color: "#2563eb",
    fontWeight: "600",
  },
  textLinkPressed: { opacity: 0.6 },
  actionSep: { fontSize: 14, color: "#d1d5db" },
  dangerAction: { fontSize: 14, color: "#b91c1c", fontWeight: "600" },
  renameRow: { gap: 10 },
  renameActions: {
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "center",
    gap: 10,
  },
  renameInput: { marginBottom: 0 },
  smallBtn: {
    backgroundColor: "#111827",
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 10,
  },
  smallBtnPressed: { opacity: 0.85 },
  smallBtnText: { color: "#fff", fontWeight: "600", fontSize: 14 },
  smallBtnGhost: {
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 10,
    backgroundColor: "#f3f4f6",
  },
  smallBtnGhostPressed: { backgroundColor: "#e5e7eb" },
  smallBtnGhostText: { color: "#374151", fontWeight: "600", fontSize: 14 },
  backLink: {
    marginTop: 28,
    alignSelf: "center",
    paddingVertical: 12,
    paddingHorizontal: 20,
  },
  backLinkPressed: { opacity: 0.55 },
  backLinkText: {
    fontSize: 15,
    color: "#6b7280",
    fontWeight: "600",
  },
});
