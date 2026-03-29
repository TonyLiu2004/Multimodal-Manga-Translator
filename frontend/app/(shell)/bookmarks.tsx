import React, { useCallback, useEffect, useState } from "react";
import {
  ActivityIndicator,
  Alert,
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
import {
  createReadingList,
  deleteReadingList,
  fetchReadingLists,
  renameReadingList,
  type ReadingListCollection,
} from "@/lib/readingListApi";

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

export default function BookmarksScreen() {
  const router = useRouter();
  const { session, loading: authLoading } = useAuth();
  const [collections, setCollections] = useState<ReadingListCollection[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [newListName, setNewListName] = useState("");
  const [creating, setCreating] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editName, setEditName] = useState("");
  const [savingRename, setSavingRename] = useState(false);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const load = useCallback(async () => {
    if (!session?.access_token) return;
    setError(null);
    try {
      const cols = await fetchReadingLists(session.access_token);
      setCollections(cols);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Could not load reading lists.",
      );
      setCollections([]);
    } finally {
      setLoading(false);
    }
  }, [session?.access_token]);

  useEffect(() => {
    if (authLoading) return;
    if (!session) {
      router.replace("/sign-in");
      return;
    }
    if (!session.access_token) return;
    setLoading(true);
    void load();
  }, [authLoading, session, load, router]);

  const onRefresh = useCallback(async () => {
    if (!session?.access_token) return;
    setRefreshing(true);
    try {
      await load();
    } finally {
      setRefreshing(false);
    }
  }, [session?.access_token, load]);

  const onCreateList = async () => {
    const name = newListName.trim();
    if (!name || !session?.access_token) return;
    setCreating(true);
    setError(null);
    try {
      await createReadingList(session.access_token, name);
      setNewListName("");
      await load();
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Could not create the reading list.",
      );
    } finally {
      setCreating(false);
    }
  };

  const startRename = (c: ReadingListCollection) => {
    setEditingId(c.id);
    setEditName(c.name);
  };

  const cancelRename = () => {
    setEditingId(null);
    setEditName("");
  };

  const onSaveRename = async () => {
    if (editingId == null || !session?.access_token) return;
    const name = editName.trim();
    if (!name) return;
    setSavingRename(true);
    setError(null);
    try {
      await renameReadingList(session.access_token, editingId, name);
      setEditingId(null);
      setEditName("");
      await load();
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Could not rename the reading list.",
      );
    } finally {
      setSavingRename(false);
    }
  };

  const confirmDelete = (c: ReadingListCollection) => {
    Alert.alert(
      "Delete list",
      `Remove “${c.name}” and every title saved in it?`,
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete",
          style: "destructive",
          onPress: () => void deleteList(c.id),
        },
      ],
    );
  };

  const deleteList = async (id: number) => {
    if (!session?.access_token) return;
    setDeletingId(id);
    setError(null);
    try {
      await deleteReadingList(session.access_token, id);
      if (editingId === id) cancelRename();
      await load();
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Could not delete the reading list.",
      );
    } finally {
      setDeletingId(null);
    }
  };

  if (authLoading || !session) {
    return (
      <SafeAreaView style={styles.safe} edges={["top", "right", "bottom"]}>
        <ActivityIndicator style={styles.loader} color="#374151" />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safe} edges={["top", "right", "bottom"]}>
      <ScrollView
        contentContainerStyle={styles.scroll}
        keyboardShouldPersistTaps="handled"
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor="#374151"
            colors={["#111827"]}
          />
        }
      >
        <View style={styles.column}>
          <Text style={styles.title}>Bookmarks</Text>
          <Text style={styles.hint}>
            Reading lists and saved manga. Create a list below, then add titles
            from search or when you open a manga.
          </Text>

          <View style={styles.newListCard}>
            <Text style={styles.newListLabel}>New list</Text>
            <TextInput
              style={styles.newListInput}
              placeholder="List name"
              placeholderTextColor="#9ca3af"
              value={newListName}
              onChangeText={setNewListName}
              editable={!creating}
              onSubmitEditing={() => void onCreateList()}
              returnKeyType="done"
            />
            <Pressable
              style={({ pressed }) => [
                styles.createButton,
                creating && styles.createButtonDisabled,
                pressed && !creating && styles.createButtonPressed,
              ]}
              onPress={() => void onCreateList()}
              disabled={
                creating || !newListName.trim() || !session.access_token
              }
            >
              {creating ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.createButtonText}>Create</Text>
              )}
            </Pressable>
          </View>

          {error ? (
            <View style={styles.noticeError}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          ) : null}

          {loading && collections.length === 0 ? (
            <ActivityIndicator style={styles.spinner} color="#374151" />
          ) : null}

          {!loading && collections.length === 0 && !error ? (
            <View style={styles.emptyCard}>
              <Text style={styles.emptyText}>
                No lists yet. Add one above, then save manga from the home screen
                or search.
              </Text>
            </View>
          ) : null}

          {collections.map((c) => (
            <View key={c.id} style={styles.card}>
              <Pressable
                style={({ pressed }) => [
                  styles.cardMain,
                  pressed && styles.cardMainPressed,
                ]}
                onPress={() =>
                  router.push({
                    pathname: "/reading-list/[id]",
                    params: { id: String(c.id), title: c.name },
                  })
                }
                disabled={deletingId === c.id}
              >
                <View style={styles.cardText}>
                  <Text style={styles.cardTitle} numberOfLines={2}>
                    {c.name}
                  </Text>
                  <Text style={styles.cardMeta}>
                    {c.manga_count} title{c.manga_count === 1 ? "" : "s"}
                  </Text>
                </View>
                <View style={styles.countPill}>
                  <Text style={styles.countPillText}>{c.manga_count}</Text>
                </View>
                <Text style={styles.openArrow}>→</Text>
              </Pressable>

              {editingId === c.id ? (
                <View style={styles.editBlock}>
                  <TextInput
                    style={styles.renameInput}
                    value={editName}
                    onChangeText={setEditName}
                    autoFocus
                    editable={!savingRename}
                  />
                  <View style={styles.editActions}>
                    <Pressable
                      style={({ pressed }) => [
                        styles.smallButton,
                        pressed && styles.smallButtonPressed,
                      ]}
                      onPress={() => void onSaveRename()}
                      disabled={savingRename || !editName.trim()}
                    >
                      {savingRename ? (
                        <ActivityIndicator color="#fff" size="small" />
                      ) : (
                        <Text style={styles.smallButtonText}>Save</Text>
                      )}
                    </Pressable>
                    <Pressable
                      style={({ pressed }) => [
                        styles.smallButtonGhost,
                        pressed && styles.smallButtonGhostPressed,
                      ]}
                      onPress={cancelRename}
                      disabled={savingRename}
                    >
                      <Text style={styles.smallButtonGhostText}>Cancel</Text>
                    </Pressable>
                  </View>
                </View>
              ) : (
                <View style={styles.cardActions}>
                  <Pressable
                    onPress={() => startRename(c)}
                    disabled={deletingId === c.id}
                    hitSlop={8}
                  >
                    <Text style={styles.actionLink}>Rename</Text>
                  </Pressable>
                  <Pressable
                    onPress={() => confirmDelete(c)}
                    disabled={deletingId === c.id}
                    hitSlop={8}
                  >
                    <Text style={styles.actionDanger}>
                      {deletingId === c.id ? "Deleting…" : "Delete"}
                    </Text>
                  </Pressable>
                </View>
              )}
            </View>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#f3f4f6" },
  loader: { marginTop: 48 },
  scroll: {
    paddingVertical: 24,
    paddingBottom: 40,
    alignItems: "center",
  },
  column: {
    width: "100%",
    maxWidth: 440,
    paddingHorizontal: 24,
    alignItems: "center",
  },
  title: {
    fontSize: 28,
    fontWeight: "800",
    color: "#111827",
    marginBottom: 8,
    letterSpacing: -0.4,
    textAlign: "center",
    alignSelf: "stretch",
  },
  hint: {
    fontSize: 14,
    color: "#6b7280",
    lineHeight: 20,
    textAlign: "center",
    marginBottom: 16,
    alignSelf: "stretch",
  },
  newListCard: {
    width: "100%",
    backgroundColor: "#fff",
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    ...cardShadow,
  },
  newListLabel: {
    fontSize: 12,
    fontWeight: "700",
    color: "#4b5563",
    textTransform: "uppercase",
    letterSpacing: 0.4,
    marginBottom: 8,
    textAlign: "center",
  },
  newListInput: {
    height: 48,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    borderRadius: 12,
    paddingHorizontal: 14,
    fontSize: 16,
    color: "#111827",
    backgroundColor: "#f9fafb",
    marginBottom: 10,
  },
  createButton: {
    backgroundColor: "#111827",
    height: 46,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
  },
  createButtonPressed: { opacity: 0.88 },
  createButtonDisabled: { opacity: 0.5 },
  createButtonText: { color: "#fff", fontSize: 15, fontWeight: "600" },
  noticeError: {
    backgroundColor: "#fef2f2",
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#fecaca",
    width: "100%",
  },
  errorText: { color: "#b91c1c", fontSize: 14 },
  spinner: { marginVertical: 24 },
  emptyCard: {
    backgroundColor: "#fff",
    borderRadius: 16,
    padding: 22,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    borderStyle: "dashed",
    width: "100%",
    ...cardShadow,
  },
  emptyText: {
    fontSize: 15,
    color: "#6b7280",
    textAlign: "center",
    lineHeight: 22,
  },
  card: {
    width: "100%",
    backgroundColor: "#fff",
    borderRadius: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    overflow: "hidden",
    ...cardShadow,
  },
  cardMain: {
    flexDirection: "row",
    alignItems: "center",
    padding: 14,
    gap: 12,
  },
  cardMainPressed: { opacity: 0.9 },
  cardText: { flex: 1, minWidth: 0 },
  cardTitle: {
    fontSize: 17,
    fontWeight: "700",
    color: "#111827",
  },
  cardMeta: {
    fontSize: 13,
    color: "#6b7280",
    marginTop: 4,
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
  openArrow: {
    fontSize: 18,
    color: "#2563eb",
    fontWeight: "600",
  },
  cardActions: {
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: 20,
    paddingHorizontal: 14,
    paddingBottom: 12,
    paddingTop: 0,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "#f3f4f6",
  },
  actionLink: {
    fontSize: 14,
    fontWeight: "600",
    color: "#2563eb",
  },
  actionDanger: {
    fontSize: 14,
    fontWeight: "600",
    color: "#b91c1c",
  },
  editBlock: {
    paddingHorizontal: 14,
    paddingBottom: 12,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "#f3f4f6",
    gap: 8,
  },
  renameInput: {
    height: 44,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    borderRadius: 10,
    paddingHorizontal: 12,
    fontSize: 16,
    color: "#111827",
    backgroundColor: "#f9fafb",
    marginTop: 8,
  },
  editActions: {
    flexDirection: "row",
    gap: 10,
    justifyContent: "flex-end",
  },
  smallButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: "#111827",
    minWidth: 72,
    alignItems: "center",
    justifyContent: "center",
  },
  smallButtonPressed: { opacity: 0.88 },
  smallButtonText: { color: "#fff", fontSize: 14, fontWeight: "600" },
  smallButtonGhost: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    backgroundColor: "#fff",
  },
  smallButtonGhostPressed: { opacity: 0.85 },
  smallButtonGhostText: { color: "#374151", fontSize: 14, fontWeight: "600" },
});
