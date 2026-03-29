import React, { useCallback, useEffect, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
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
  fetchReadingListItems,
  fetchReadingLists,
  removeReadingListItem,
  renameReadingList,
  type ReadingListCollection,
  type ReadingListItem,
} from "@/lib/readingListApi";
import { fetchMangaCoverUrl } from "@/lib/mangaCoverApi";

const COVER_PLACEHOLDER =
  "https://via.placeholder.com/128x180?text=No+Cover";

async function hydrateMangaCovers(
  items: ReadingListItem[],
  setCoverUrls: React.Dispatch<
    React.SetStateAction<Record<string, string | null>>
  >,
) {
  const extIds = [
    ...new Set(
      items.map((i) => i.external_manga_id).filter(Boolean),
    ),
  ] as string[];
  setCoverUrls((prev) => {
    const need = extIds.filter((id) => !(id in prev));
    if (need.length === 0) return prev;
    void (async () => {
      const results = Object.fromEntries(
        await Promise.all(
          need.map(async (id) => [id, await fetchMangaCoverUrl(id)] as const),
        ),
      ) as Record<string, string | null>;
      setCoverUrls((p) => ({ ...p, ...results }));
    })();
    return prev;
  });
}

export default function ProfileScreen() {
  const router = useRouter();
  const { session, loading: authLoading } = useAuth();
  const [displayName, setDisplayName] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  const [collections, setCollections] = useState<ReadingListCollection[]>([]);
  /** Undefined = items not loaded yet (list collapsed or never opened). */
  const [itemsByListId, setItemsByListId] = useState<
    Record<number, ReadingListItem[] | undefined>
  >({});
  const [expandedByListId, setExpandedByListId] = useState<
    Record<number, boolean>
  >({});
  const [itemsLoadingByListId, setItemsLoadingByListId] = useState<
    Record<number, boolean>
  >({});
  const [coverUrls, setCoverUrls] = useState<Record<string, string | null>>({});
  const [listLoading, setListLoading] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [removingKey, setRemovingKey] = useState<string | null>(null);
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
      setItemsByListId({});
      setExpandedByListId({});
      setItemsLoadingByListId({});
    } catch (e) {
      setListError(
        e instanceof Error ? e.message : "Could not load reading lists.",
      );
      setCollections([]);
      setItemsByListId({});
      setExpandedByListId({});
    } finally {
      setListLoading(false);
    }
  }, [session?.access_token]);

  const loadListItems = useCallback(
    async (listId: number) => {
      if (!session?.access_token) return;
      setItemsLoadingByListId((prev) => ({ ...prev, [listId]: true }));
      setListError(null);
      try {
        const items = await fetchReadingListItems(
          session.access_token,
          listId,
        );
        setItemsByListId((prev) => ({ ...prev, [listId]: items }));
        await hydrateMangaCovers(items, setCoverUrls);
      } catch (e) {
        setListError(
          e instanceof Error ? e.message : "Could not load list items.",
        );
      } finally {
        setItemsLoadingByListId((prev) => ({ ...prev, [listId]: false }));
      }
    },
    [session?.access_token],
  );

  const toggleListExpanded = useCallback(
    (listId: number) => {
      setExpandedByListId((prev) => {
        const nextOpen = !prev[listId];
        if (nextOpen) {
          queueMicrotask(() => {
            setItemsByListId((itemsPrev) => {
              if (itemsPrev[listId] === undefined) {
                void loadListItems(listId);
              }
              return itemsPrev;
            });
          });
        }
        return { ...prev, [listId]: nextOpen };
      });
    },
    [loadListItems],
  );

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
      setItemsByListId((prev) => {
        const next = { ...prev };
        delete next[listId];
        return next;
      });
      setExpandedByListId((prev) => {
        const next = { ...prev };
        delete next[listId];
        return next;
      });
      setItemsLoadingByListId((prev) => {
        const next = { ...prev };
        delete next[listId];
        return next;
      });
      if (editingListId === listId) setEditingListId(null);
    } catch (e) {
      setListError(
        e instanceof Error ? e.message : "Could not delete reading list.",
      );
    }
  };

  const onRemoveItem = async (listId: number, mangaId: number) => {
    if (!session?.access_token) return;
    const key = `${listId}-${mangaId}`;
    setRemovingKey(key);
    setListError(null);
    try {
      await removeReadingListItem(session.access_token, listId, mangaId);
      setItemsByListId((prev) => ({
        ...prev,
        [listId]: (prev[listId] ?? []).filter((x) => x.manga_id !== mangaId),
      }));
      setCollections((prev) =>
        prev.map((c) =>
          c.id === listId
            ? { ...c, manga_count: Math.max(0, c.manga_count - 1) }
            : c,
        ),
      );
    } catch (e) {
      setListError(
        e instanceof Error ? e.message : "Could not remove from list.",
      );
    } finally {
      setRemovingKey(null);
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
              Tap a list to see titles. Add manga from search or the home
              carousel.
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
            const expanded = !!expandedByListId[c.id];
            const itemsLoading = !!itemsLoadingByListId[c.id];
            const items = itemsByListId[c.id];
            const itemsLoaded = items !== undefined;
            return (
              <View key={c.id} style={styles.listCard}>
                <View style={styles.listCardHeader}>
                  {isEditing ? (
                    <View style={styles.renameRow}>
                      <TextInput
                        style={[styles.input, styles.renameInput]}
                        value={editNameDraft}
                        onChangeText={setEditNameDraft}
                        autoFocus
                      />
                      <Pressable
                        style={styles.smallBtn}
                        onPress={() => void onSaveRename(c.id)}
                        disabled={savingRename || !editNameDraft.trim()}
                      >
                        <Text style={styles.smallBtnText}>Save</Text>
                      </Pressable>
                      <Pressable
                        style={styles.smallBtnMuted}
                        onPress={() => setEditingListId(null)}
                      >
                        <Text style={styles.smallBtnMutedText}>Cancel</Text>
                      </Pressable>
                    </View>
                  ) : (
                    <Pressable
                      style={styles.listHeaderPressable}
                      onPress={() => toggleListExpanded(c.id)}
                      accessibilityRole="button"
                      accessibilityLabel={`${expanded ? "Collapse" : "Expand"} list ${c.name}`}
                    >
                      <View style={styles.listHeaderRow}>
                        <Text style={styles.listChevron}>
                          {expanded ? "▼" : "▶"}
                        </Text>
                        <View style={styles.listHeaderTitles}>
                          <Text style={styles.listCardTitle} numberOfLines={2}>
                            {c.name}
                          </Text>
                          <Text style={styles.listCardCount}>
                            {c.manga_count} title
                            {c.manga_count === 1 ? "" : "s"}
                          </Text>
                        </View>
                      </View>
                    </Pressable>
                  )}
                </View>
                {!isEditing ? (
                  <View style={styles.listCardActions}>
                    <Pressable
                      onPress={() => {
                        setEditingListId(c.id);
                        setEditNameDraft(c.name);
                      }}
                    >
                      <Text style={styles.linkAction}>Rename</Text>
                    </Pressable>
                    <Text style={styles.actionSep}>·</Text>
                    <Pressable onPress={() => confirmDeleteList(c)}>
                      <Text style={styles.dangerAction}>Delete list</Text>
                    </Pressable>
                  </View>
                ) : null}

                {expanded && !isEditing ? (
                  <>
                    {itemsLoading ? (
                      <ActivityIndicator
                        style={styles.listItemsSpinner}
                        color="#111"
                      />
                    ) : null}
                    {!itemsLoading && itemsLoaded
                      ? (items as ReadingListItem[]).map((item) => {
                          const rk = `${c.id}-${item.manga_id}`;
                          const extId = item.external_manga_id;
                          const resolvedCover =
                            extId != null && extId !== ""
                              ? coverUrls[extId]
                              : null;
                          const coverUri =
                            resolvedCover ?? COVER_PLACEHOLDER;
                          return (
                            <View key={item.id} style={styles.listRow}>
                              <Image
                                source={{ uri: coverUri }}
                                style={styles.listRowThumb}
                              />
                              <View style={styles.listRowText}>
                                <Text
                                  style={styles.listTitle}
                                  numberOfLines={2}
                                >
                                  {item.manga_title}
                                </Text>
                                {item.last_chapter_number != null ? (
                                  <Text style={styles.listMeta}>
                                    Last read: Ch.{" "}
                                    {item.last_chapter_number}
                                  </Text>
                                ) : null}
                              </View>
                              <Pressable
                                style={styles.removeBtn}
                                onPress={() =>
                                  void onRemoveItem(c.id, item.manga_id)
                                }
                                disabled={removingKey === rk}
                              >
                                {removingKey === rk ? (
                                  <ActivityIndicator
                                    size="small"
                                    color="#c62828"
                                  />
                                ) : (
                                  <Text style={styles.removeBtnText}>
                                    Remove
                                  </Text>
                                )}
                              </Pressable>
                            </View>
                          );
                        })
                      : null}
                    {!itemsLoading &&
                    itemsLoaded &&
                    (items as ReadingListItem[]).length === 0 ? (
                      <Text style={styles.emptyItems}>
                        No manga in this list.
                      </Text>
                    ) : null}
                  </>
                ) : null}
              </View>
            );
          })}

          <Pressable onPress={() => router.back()} style={styles.linkWrap}>
            <Text style={styles.muted}>Back</Text>
          </Pressable>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#fff" },
  flex: { flex: 1 },
  loader: { marginTop: 40 },
  scroll: {
    padding: 24,
    alignItems: "stretch",
    maxWidth: 420,
    width: "100%",
    alignSelf: "center",
  },
  title: {
    fontSize: 28,
    fontWeight: "bold",
    marginBottom: 8,
    color: "#111",
  },
  emailLine: {
    fontSize: 14,
    color: "#666",
    marginBottom: 24,
  },
  fieldLabel: {
    fontSize: 14,
    color: "#333",
    marginBottom: 8,
    fontWeight: "500",
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
  muted: { fontSize: 14, color: "#888" },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: "#111",
    marginTop: 28,
    marginBottom: 6,
  },
  sectionHint: {
    fontSize: 13,
    color: "#777",
    marginBottom: 14,
    lineHeight: 18,
  },
  newListRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    marginBottom: 8,
  },
  newListInput: { flex: 1, marginBottom: 0 },
  secondaryBtn: {
    height: 50,
    paddingHorizontal: 16,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#111",
    alignItems: "center",
    justifyContent: "center",
    minWidth: 88,
  },
  secondaryBtnText: { fontSize: 15, fontWeight: "600", color: "#111" },
  listSpinner: { marginVertical: 16 },
  listItemsSpinner: { marginVertical: 12, alignSelf: "flex-start" },
  listError: { color: "#c62828", fontSize: 14, marginBottom: 8 },
  emptyList: { fontSize: 14, color: "#888", fontStyle: "italic" },
  listCard: {
    marginTop: 16,
    paddingBottom: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#ddd",
  },
  listCardHeader: { marginBottom: 6 },
  listHeaderPressable: {
    paddingVertical: 4,
    marginHorizontal: -4,
    paddingHorizontal: 4,
  },
  listHeaderRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  listChevron: {
    fontSize: 12,
    color: "#666",
    width: 18,
  },
  listHeaderTitles: { flex: 1, minWidth: 0 },
  listCardTitle: { fontSize: 17, fontWeight: "700", color: "#111" },
  listCardCount: { fontSize: 13, color: "#666", marginTop: 2 },
  listCardActions: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 10,
    gap: 6,
  },
  linkAction: { fontSize: 14, color: "#1565c0", fontWeight: "500" },
  actionSep: { fontSize: 14, color: "#999" },
  dangerAction: { fontSize: 14, color: "#c62828", fontWeight: "500" },
  renameRow: { gap: 8 },
  renameInput: { marginBottom: 0 },
  smallBtn: {
    alignSelf: "flex-start",
    backgroundColor: "#111",
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 8,
  },
  smallBtnText: { color: "#fff", fontWeight: "600", fontSize: 14 },
  smallBtnMuted: { alignSelf: "flex-start", paddingVertical: 8 },
  smallBtnMutedText: { color: "#666", fontSize: 14 },
  listRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 10,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "#eee",
    gap: 12,
  },
  listRowThumb: {
    width: 48,
    height: 68,
    borderRadius: 6,
    backgroundColor: "#eee",
  },
  listRowText: { flex: 1, minWidth: 0 },
  listTitle: { fontSize: 15, color: "#222", fontWeight: "500" },
  listMeta: { fontSize: 12, color: "#666", marginTop: 4 },
  removeBtn: { paddingVertical: 8, paddingHorizontal: 10, minWidth: 72 },
  removeBtnText: { fontSize: 14, color: "#c62828", fontWeight: "600" },
  emptyItems: {
    fontSize: 13,
    color: "#999",
    fontStyle: "italic",
    paddingVertical: 8,
  },
});
