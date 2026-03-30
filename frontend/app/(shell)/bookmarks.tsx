import React, { useCallback, useEffect, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { useRouter } from "expo-router";
import { SafeAreaView } from "react-native-safe-area-context";
import BookmarksCollectionTile, {
  resolveCollectionCoverUri,
} from "@/app/(shell)/bookmarks/BookmarksCollectionTile";
import BookmarksNewListTile from "@/app/(shell)/bookmarks/BookmarksNewListTile";
import CreateReadingListModal from "@/app/(shell)/bookmarks/CreateReadingListModal";
import { useAuth } from "@/context/AuthContext";
import { fetchMangaCoverUrl } from "@/lib/mangaCoverApi";
import {
  createReadingList,
  deleteReadingList,
  fetchReadingLists,
  renameReadingList,
  type ReadingListCollection,
} from "@/lib/readingListApi";

const gridGap = 8;

export default function BookmarksScreen() {
  const router = useRouter();
  const { session, loading: authLoading } = useAuth();

  const [collections, setCollections] = useState<ReadingListCollection[]>([]);
  const [coverUrls, setCoverUrls] = useState<Record<string, string | null>>(
    {},
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const [newListName, setNewListName] = useState("");
  const [creating, setCreating] = useState(false);
  const [createModalVisible, setCreateModalVisible] = useState(false);

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

  useEffect(() => {
    const extIds = [
      ...new Set(
        collections
          .map((c) => c.latest_external_manga_id)
          .filter((x): x is string => Boolean(x)),
      ),
    ];
    if (extIds.length === 0) return;

    let cancelled = false;
    void (async () => {
      const entries = await Promise.all(
        extIds.map(async (id) => [id, await fetchMangaCoverUrl(id)] as const),
      );
      if (cancelled) return;
      setCoverUrls((prev) => ({ ...prev, ...Object.fromEntries(entries) }));
    })();
    return () => {
      cancelled = true;
    };
  }, [collections]);

  const onRefresh = useCallback(async () => {
    if (!session?.access_token) return;
    setRefreshing(true);
    try {
      await load();
    } finally {
      setRefreshing(false);
    }
  }, [session?.access_token, load]);

  const closeCreateModal = () => {
    if (creating) return;
    setCreateModalVisible(false);
    setNewListName("");
  };

  const openCreateModal = () => {
    setNewListName("");
    setCreateModalVisible(true);
  };

  const onCreateList = async () => {
    const name = newListName.trim();
    if (!name || !session?.access_token) return;
    setCreating(true);
    setError(null);
    try {
      await createReadingList(session.access_token, name);
      setNewListName("");
      setCreateModalVisible(false);
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

  const createSubmitEnabled = Boolean(
    session.access_token && newListName.trim(),
  );

  return (
    <SafeAreaView style={styles.safe} edges={["top", "right", "bottom"]}>
      <CreateReadingListModal
        visible={createModalVisible}
        creating={creating}
        newListName={newListName}
        onChangeName={setNewListName}
        onClose={closeCreateModal}
        onSubmit={onCreateList}
        submitEnabled={createSubmitEnabled}
      />

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
            Reading lists and saved manga. Use Create new list to add one, then
            save titles from search or when you open a manga.
          </Text>

          {error ? (
            <View style={styles.noticeError}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          ) : null}

          {loading && collections.length === 0 ? (
            <ActivityIndicator style={styles.spinner} color="#374151" />
          ) : (
            <View style={[styles.gridWrap, { gap: gridGap }]}>
              <BookmarksNewListTile
                creating={creating}
                onPressCreate={openCreateModal}
              />

              {collections.map((c) => {
                const ext = c.latest_external_manga_id;
                const { uri, loading: coverLoading } =
                  resolveCollectionCoverUri(ext, coverUrls);
                return (
                  <BookmarksCollectionTile
                    key={c.id}
                    collection={c}
                    coverUri={uri}
                    coverLoading={coverLoading}
                    isEditing={editingId === c.id}
                    editName={editName}
                    onChangeEditName={setEditName}
                    onOpenList={() =>
                      router.push({
                        pathname: "/reading-list/[id]",
                        params: { id: String(c.id), title: c.name },
                      })
                    }
                    onStartRename={() => startRename(c)}
                    onCancelRename={cancelRename}
                    onSaveRename={onSaveRename}
                    savingRename={savingRename}
                    onRequestDelete={() => confirmDelete(c)}
                    deleting={deletingId === c.id}
                  />
                );
              })}
            </View>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#f3f4f6" },
  loader: { marginTop: 48 },
  scroll: {
    flexGrow: 1,
    width: "100%",
    paddingVertical: 24,
    paddingBottom: 40,
    paddingHorizontal: 20,
    alignItems: "stretch",
  },
  column: {
    width: "100%",
    alignItems: "stretch",
  },
  title: {
    fontSize: 28,
    fontWeight: "800",
    color: "#111827",
    marginBottom: 8,
    letterSpacing: -0.4,
    textAlign: "left",
  },
  hint: {
    fontSize: 14,
    color: "#6b7280",
    lineHeight: 20,
    textAlign: "left",
    marginBottom: 16,
  },
  noticeError: {
    backgroundColor: "#fef2f2",
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#fecaca",
    width: "100%",
  },
  errorText: { color: "#b91c1c", fontSize: 14, textAlign: "left" },
  spinner: { marginVertical: 24 },
  gridWrap: {
    flexDirection: "row",
    flexWrap: "wrap",
    width: "100%",
    marginBottom: 12,
    alignItems: "flex-start",
  },
});
