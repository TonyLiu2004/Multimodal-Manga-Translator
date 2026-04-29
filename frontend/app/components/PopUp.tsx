import { type Href, useRouter } from "expo-router";
import {
  Modal,
  Text,
  ScrollView,
  View,
  StyleSheet,
  Pressable,
  Image,
  ActivityIndicator,
  TextInput,
} from "react-native";
import React from "react";
import { Chapter, Manga } from "@/lib/mangaTypes";
import { parseChapterNumber } from "@/lib/readingListDetailManga";
import { useAuth } from "@/context/AuthContext";
import {
  addToReadingList,
  createReadingList,
  fetchReadingLists,
  type ReadingListCollection,
} from "@/lib/readingListApi";

interface PopUpProps {
  visible: boolean;
  title: string;
  summary: string;
  coverArt: string;
  manga: Manga;
  chapters: Chapter[];
  loadingChapters: boolean;
  onClose: () => void;
  /** When opened from a reading list, pass this so reader can PATCH last-read chapter. */
  readingListProgress?: { readingListId: number; mangaId: number };
}

export default function PopUp({
  visible,
  title,
  summary,
  coverArt,
  chapters,
  loadingChapters,
  manga,
  onClose,
  readingListProgress,
}: PopUpProps) {
  const router = useRouter();
  const { session, loading: authLoading } = useAuth();
  const [listBusy, setListBusy] = React.useState(false);
  const [listMsg, setListMsg] = React.useState<string | null>(null);
  const [readingLists, setReadingLists] = React.useState<
    ReadingListCollection[]
  >([]);
  const [listsLoading, setListsLoading] = React.useState(false);
  const [selectedListIds, setSelectedListIds] = React.useState<Set<number>>(
    () => new Set(),
  );
  const [createListOpen, setCreateListOpen] = React.useState(false);
  const [createListName, setCreateListName] = React.useState("");
  const [createListBusy, setCreateListBusy] = React.useState(false);
  const [createListMsg, setCreateListMsg] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (visible) {
      setListMsg(null);
      setListBusy(false);
      setCreateListOpen(false);
      setCreateListName("");
      setCreateListBusy(false);
      setCreateListMsg(null);
    }
  }, [visible, manga.id]);

  React.useEffect(() => {
    if (!visible || !session?.access_token) return;
    let cancelled = false;
    (async () => {
      setListsLoading(true);
      try {
        let cols = await fetchReadingLists(session.access_token);
        if (!cancelled && cols.length === 0) {
          await createReadingList(session.access_token, "My list");
          cols = await fetchReadingLists(session.access_token);
        }
        if (cancelled) return;
        setReadingLists(cols);
        setSelectedListIds((prev) => {
          const next = new Set<number>();
          // Keep any still-valid selections
          for (const id of prev) {
            if (cols.some((c) => c.id === id)) next.add(id);
          }
          // Default to first list if nothing selected
          if (next.size === 0 && cols[0]?.id != null) next.add(cols[0].id);
          return next;
        });
      } catch {
        if (!cancelled) {
          setReadingLists([]);
          setSelectedListIds(new Set());
        }
      } finally {
        if (!cancelled) setListsLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [visible, session?.access_token]);

  const [selectedLanguage, setSelectedLanguage] = React.useState<string>("All");
  const filteredChapters = chapters.filter(
    (ch) => selectedLanguage === "All" || ch.language === selectedLanguage,
  );

  const onCreateList = async () => {
    if (!session?.access_token) return;
    const name = createListName.trim();
    if (!name) {
      setCreateListMsg("Please enter a name.");
      return;
    }
    setCreateListBusy(true);
    setCreateListMsg(null);
    try {
      await createReadingList(session.access_token, name);
      const cols = await fetchReadingLists(session.access_token);
      setReadingLists(cols);
      const created = cols.find((c) => c.name === name) ?? cols[0];
      setSelectedListIds((prev) => {
        const next = new Set(prev);
        if (created?.id != null) next.add(created.id);
        return next;
      });
      setCreateListOpen(false);
      setCreateListName("");
      setCreateListMsg("Created.");
    } catch (e) {
      setCreateListMsg(
        e instanceof Error ? e.message : "Could not create list.",
      );
    } finally {
      setCreateListBusy(false);
    }
  };

  return (
    <Modal
      visible={visible}
      transparent={true}
      animationType="fade"
      onRequestClose={onClose}
    >
      <Pressable style={styles.overlay} onPress={onClose}>
        <Pressable style={styles.popup} onPress={(e) => e.stopPropagation()}>
          <ScrollView style={styles.content}>
            <Image source={{ uri: coverArt }} style={styles.coverImage} />
            <Text style={styles.title}>{title}</Text>
            <Text style={styles.summary}>{summary}</Text>

            <View style={styles.card}>
              {session?.access_token ? (
              <View style={styles.readingListBlock}>
                <View style={styles.listHeaderRow}>
                  <Text style={styles.listPickerLabel}>Add to list</Text>
                  <Pressable
                    style={({ pressed }) => [
                      styles.createListSmallBtn,
                      pressed && styles.createListSmallBtnPressed,
                    ]}
                    onPress={() => {
                      setCreateListMsg(null);
                      setCreateListName("");
                      setCreateListOpen(true);
                    }}
                    disabled={listsLoading || createListBusy}
                  >
                    <Text style={styles.createListSmallBtnText}>New list</Text>
                  </Pressable>
                </View>
                {listsLoading ? (
                  <ActivityIndicator
                    size="small"
                    color="#111"
                    style={styles.listsSpinner}
                  />
                ) : readingLists.length === 0 ? (
                  <Text style={styles.listHint}>
                    Open Bookmarks in the sidebar to create a list, then try
                    again.
                  </Text>
                ) : (
                  <View style={styles.listChipsWrap}>
                    {readingLists.map((c) => (
                      <Pressable
                        key={c.id}
                        style={[
                          styles.listChip,
                          selectedListIds.has(c.id) && styles.listChipActive,
                        ]}
                        onPress={() =>
                          setSelectedListIds((prev) => {
                            const next = new Set(prev);
                            if (next.has(c.id)) next.delete(c.id);
                            else next.add(c.id);
                            return next;
                          })
                        }
                      >
                        <Text
                          style={[
                            styles.listChipText,
                            selectedListIds.has(c.id) &&
                              styles.listChipTextActive,
                          ]}
                          numberOfLines={1}
                        >
                          {c.name}
                        </Text>
                      </Pressable>
                    ))}
                  </View>
                )}
                <Pressable
                  style={[
                    styles.addListBtn,
                    (listBusy || selectedListIds.size === 0 || listsLoading) &&
                      styles.addListBtnDisabled,
                  ]}
                  onPress={async () => {
                    if (!session.access_token || selectedListIds.size === 0)
                      return;
                    setListMsg(null);
                    setListBusy(true);
                    try {
                      const ids = Array.from(selectedListIds);
                      const results = await Promise.allSettled(
                        ids.map((readingListId) =>
                          addToReadingList(session.access_token, {
                            readingListId,
                            external_manga_id: manga.id,
                            manga_title: title,
                          }),
                        ),
                      );
                      const failures = results.filter(
                        (r) => r.status === "rejected",
                      ) as PromiseRejectedResult[];
                      if (failures.length > 0) {
                        const first = failures[0]?.reason;
                        setListMsg(
                          first instanceof Error
                            ? first.message
                            : "Could not add to one or more lists.",
                        );
                      } else {
                        setListMsg(
                          `Added to ${ids.length} list${ids.length === 1 ? "" : "s"}.`,
                        );
                      }
                    } catch (e) {
                      setListMsg(
                        e instanceof Error
                          ? e.message
                          : "Could not add to list.",
                      );
                    } finally {
                      setListBusy(false);
                    }
                  }}
                  disabled={
                    listBusy || selectedListIds.size === 0 || listsLoading
                  }
                >
                  {listBusy ? (
                    <ActivityIndicator color="#fff" />
                  ) : (
                    <Text style={styles.addListBtnText}>
                      Add to selected lists
                    </Text>
                  )}
                </Pressable>
                {listMsg ? (
                  <Text
                    style={[
                      styles.listMsg,
                      listMsg.startsWith("Added")
                        ? styles.listMsgOk
                        : styles.listMsgErr,
                    ]}
                  >
                    {listMsg}
                  </Text>
                ) : null}

                {createListMsg ? (
                  <Text
                    style={[
                      styles.listMsg,
                      createListMsg === "Created."
                        ? styles.listMsgOk
                        : styles.listMsgErr,
                    ]}
                  >
                    {createListMsg}
                  </Text>
                ) : null}
              </View>
            ) : authLoading ? (
              <View style={styles.readingListBlock}>
                <Text style={styles.listPickerLabel}>Add to list</Text>
                <ActivityIndicator
                  size="small"
                  color="#111"
                  style={styles.listsSpinner}
                />
              </View>
            ) : (
              <View style={styles.readingListBlock}>
                <Text style={styles.listPickerLabel}>Add to list</Text>
                <Text style={styles.listHint}>
                  Sign in to save this title to your reading lists.
                </Text>
                <Pressable
                  style={({ pressed }) => [
                    styles.signInCtaBtn,
                    pressed && styles.signInCtaBtnPressed,
                  ]}
                  onPress={() => {
                    onClose();
                    router.push("/sign-in" as Href);
                  }}
                >
                  <Text style={styles.signInCtaBtnText}>Sign in</Text>
                </Pressable>
              </View>
            )}</View>

            <Modal
              visible={createListOpen}
              transparent
              animationType="fade"
              onRequestClose={() =>
                createListBusy ? null : setCreateListOpen(false)
              }
            >
              <Pressable
                style={styles.miniOverlay}
                onPress={() => (createListBusy ? null : setCreateListOpen(false))}
              >
                <Pressable
                  style={styles.miniCard}
                  onPress={(e) => e.stopPropagation()}
                >
                  <Text style={styles.miniTitle}>Create new list</Text>
                  <TextInput
                    style={styles.miniInput}
                    placeholder="List name"
                    placeholderTextColor="#9ca3af"
                    value={createListName}
                    onChangeText={setCreateListName}
                    editable={!createListBusy}
                    autoFocus
                    returnKeyType="done"
                    onSubmitEditing={() => (createListBusy ? null : onCreateList())}
                  />
                  <View style={styles.miniActions}>
                    <Pressable
                      style={({ pressed }) => [
                        styles.miniBtn,
                        styles.miniBtnSecondary,
                        pressed && !createListBusy && styles.miniBtnPressed,
                      ]}
                      onPress={() => setCreateListOpen(false)}
                      disabled={createListBusy}
                    >
                      <Text style={styles.miniBtnSecondaryText}>Cancel</Text>
                    </Pressable>
                    <Pressable
                      style={({ pressed }) => [
                        styles.miniBtn,
                        styles.miniBtnPrimary,
                        createListBusy && styles.miniBtnDisabled,
                        pressed && !createListBusy && styles.miniBtnPressed,
                      ]}
                      onPress={onCreateList}
                      disabled={createListBusy}
                    >
                      {createListBusy ? (
                        <ActivityIndicator color="#fff" />
                      ) : (
                        <Text style={styles.miniBtnPrimaryText}>Create</Text>
                      )}
                    </Pressable>
                  </View>
                </Pressable>
              </Pressable>
            </Modal>

            

            <View style={styles.chaptersSection}>
              <Text style={styles.chaptersTitle}>Chapters</Text>
              <Text style={styles.languageLabel}>Select Language:</Text>
              <View style={styles.languageContainer}>
                {["All", ...manga.attributes.availableTranslatedLanguages].map(
                  (lang) => (
                    <Pressable
                      key={lang}
                      style={[
                        styles.langChip,
                        selectedLanguage === lang && styles.activeLangChip,
                      ]}
                      onPress={() => setSelectedLanguage(lang)}
                    >
                      <Text
                        style={[
                          styles.langChipText,
                          selectedLanguage === lang &&
                            styles.activeLangChipText,
                        ]}
                      >
                        {lang.toUpperCase()}
                      </Text>
                    </Pressable>
                  ),
                )}
              </View>
              {loadingChapters ? (
                <ActivityIndicator
                  size="large"
                  color="#007AFF"
                  style={{ marginTop: 20 }}
                />
              ) : filteredChapters.length > 0 ? (
                <View style={styles.chaptersList}>
                  {filteredChapters.map((chapter) => (
                    <Pressable
                      key={chapter.id}
                      style={styles.chapterItem}
                      onPress={() => {
                        onClose();
                        const chNum = parseChapterNumber(chapter.chapter);
                        const ctx = readingListProgress;
                        const q = new URLSearchParams();
                        q.set("seriesId", manga.id);
                        if (ctx != null && chNum != null) {
                          q.set(
                            "readingListId",
                            String(ctx.readingListId),
                          );
                          q.set("mangaId", String(ctx.mangaId));
                          q.set("chapterNumber", String(chNum));
                        }
                        const qs = q.toString();
                        router.push(
                          (qs
                            ? `/reader/${chapter.id}?${qs}`
                            : `/reader/${chapter.id}`) as Href,
                        );
                      }}
                    >
                      <Text style={styles.chapterNumber}>
                        Ch. {chapter.chapter}
                      </Text>
                      <Text style={styles.chapterTitle}>
                        {chapter.title || "No title"}
                      </Text>
                      <Text style={styles.chapterPages}>
                        {chapter.pages} pages
                      </Text>
                    </Pressable>
                  ))}
                </View>
              ) : (
                <Text style={styles.noChapters}>No chapters available</Text>
              )}
            </View>
          </ScrollView>
          <Pressable style={styles.closeButton} onPress={onClose}>
            <Text style={styles.closeButtonText}>Close</Text>
          </Pressable>
        </Pressable>
      </Pressable>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    justifyContent: "center",
    alignItems: "center",
  },
  popup: {
    backgroundColor: "white",
    borderRadius: 15,
    padding: 20,
    width: "80%",
    maxWidth: 600,
    maxHeight: "80%",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 5,
  },
  content: {
    marginBottom: 15,
  },
  coverImage: {
    width: "100%",
    height: 300,
    borderRadius: 10,
    marginBottom: 15,
    resizeMode: "contain",
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 15,
    color: "#333",
  },
  summary: {
    fontSize: 16,
    lineHeight: 24,
    color: "#666",
    marginBottom: 20,
  },
  card: { 
    padding: 15, 
    backgroundColor: "#f4f4f4", 
    borderRadius: 12, 
    marginBottom: 20 
  },
  readingListBlock: {
    marginBottom: 16,
  },
  listHeaderRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 10,
    marginBottom: 8,
  },
  listPickerLabel: {
    fontSize: 14,
    fontWeight: "600",
    color: "#444",
  },
  createListSmallBtn: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "#DDD",
    backgroundColor: "#fff",
  },
  createListSmallBtnPressed: {
    opacity: 0.85,
  },
  createListSmallBtnText: {
    fontSize: 12,
    fontWeight: "700",
    color: "#111",
  },
  listsSpinner: {
    alignSelf: "flex-start",
    marginBottom: 10,
  },
  listHint: {
    fontSize: 13,
    color: "#888",
    marginBottom: 10,
    lineHeight: 18,
  },
  listChipsWrap: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginBottom: 12,
  },
  listChip: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: "#F0F0F0",
    borderWidth: 1,
    borderColor: "#DDD",
    maxWidth: "100%",
  },
  listChipActive: {
    backgroundColor: "#111",
    borderColor: "#111",
  },
  listChipText: {
    fontSize: 13,
    color: "#333",
    fontWeight: "500",
    maxWidth: 200,
  },
  listChipTextActive: {
    color: "#FFF",
  },
  addListBtn: {
    backgroundColor: "#111",
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignItems: "center",
  },
  addListBtnDisabled: {
    opacity: 0.6,
  },
  addListBtnText: {
    color: "#fff",
    fontSize: 15,
    fontWeight: "600",
  },
  signInCtaBtn: {
    backgroundColor: "#111",
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignItems: "center",
    alignSelf: "stretch",
  },
  signInCtaBtnPressed: {
    opacity: 0.88,
  },
  signInCtaBtnText: {
    color: "#fff",
    fontSize: 15,
    fontWeight: "600",
  },
  listMsg: {
    marginTop: 8,
    fontSize: 13,
  },
  listMsgOk: {
    color: "#2e7d32",
  },
  listMsgErr: {
    color: "#c62828",
  },
  miniOverlay: {
    flex: 1,
    backgroundColor: "rgba(17, 24, 39, 0.45)",
    paddingHorizontal: 22,
    justifyContent: "center",
    alignItems: "center",
  },
  miniCard: {
    width: "100%",
    maxWidth: 340,
    backgroundColor: "#fff",
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: "#e5e7eb",
  },
  miniTitle: {
    fontSize: 15,
    fontWeight: "800",
    color: "#111",
    textAlign: "center",
    marginBottom: 10,
  },
  miniInput: {
    height: 44,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    borderRadius: 12,
    paddingHorizontal: 14,
    fontSize: 15,
    color: "#111",
    backgroundColor: "#f9fafb",
  },
  miniActions: {
    flexDirection: "row",
    gap: 10,
    marginTop: 12,
  },
  miniBtn: {
    flex: 1,
    height: 42,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
  },
  miniBtnSecondary: {
    borderWidth: 1,
    borderColor: "#d1d5db",
    backgroundColor: "#fff",
  },
  miniBtnSecondaryText: {
    color: "#111",
    fontWeight: "700",
    fontSize: 14,
  },
  miniBtnPrimary: {
    backgroundColor: "#111",
  },
  miniBtnPrimaryText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: 14,
  },
  miniBtnPressed: {
    opacity: 0.88,
  },
  miniBtnDisabled: {
    opacity: 0.6,
  },
  chaptersSection: {
    marginTop: 20,
    paddingTop: 20,
    borderTopWidth: 1,
    borderTopColor: "#E0E0E0",
  },
  chaptersTitle: {
    fontSize: 20,
    fontWeight: "bold",
    marginBottom: 15,
    color: "#333",
  },
  chaptersList: {
    gap: 10,
  },
  chapterItem: {
    padding: 12,
    backgroundColor: "#F5F5F5",
    borderRadius: 8,
    marginBottom: 8,
  },
  chapterNumber: {
    fontSize: 16,
    fontWeight: "600",
    color: "#007AFF",
    marginBottom: 4,
  },
  chapterTitle: {
    fontSize: 14,
    color: "#333",
    marginBottom: 4,
  },
  chapterPages: {
    fontSize: 12,
    color: "#999",
  },
  noChapters: {
    fontSize: 14,
    color: "#999",
    fontStyle: "italic",
    textAlign: "center",
    marginTop: 10,
  },
  languageLabel: {
    fontSize: 14,
    fontWeight: "600",
    color: "#666",
    marginBottom: 10,
  },
  languageContainer: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginBottom: 20,
  },
  langChip: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    backgroundColor: "#F0F0F0",
    borderWidth: 1,
    borderColor: "#DDD",
  },
  activeLangChip: {
    backgroundColor: "#007AFF",
    borderColor: "#007AFF",
  },
  langChipText: {
    fontSize: 12,
    color: "#333",
    fontWeight: "500",
  },
  activeLangChipText: {
    color: "#FFF",
    fontWeight: "bold",
  },
  closeButton: {
    backgroundColor: "#007AFF",
    padding: 12,
    borderRadius: 8,
    alignItems: "center",
  },
  closeButtonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "600",
  },
});
