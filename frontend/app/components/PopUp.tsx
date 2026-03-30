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
} from "react-native";
import React from "react";
import { Chapter, Manga } from "../types/types";
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
}: PopUpProps) {
  const router = useRouter();
  const { session, loading: authLoading } = useAuth();
  const [listBusy, setListBusy] = React.useState(false);
  const [listMsg, setListMsg] = React.useState<string | null>(null);
  const [readingLists, setReadingLists] = React.useState<
    ReadingListCollection[]
  >([]);
  const [listsLoading, setListsLoading] = React.useState(false);
  const [selectedListId, setSelectedListId] = React.useState<number | null>(
    null,
  );

  React.useEffect(() => {
    if (visible) {
      setListMsg(null);
      setListBusy(false);
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
        setSelectedListId((prev) => {
          if (prev != null && cols.some((c) => c.id === prev)) return prev;
          return cols[0]?.id ?? null;
        });
      } catch {
        if (!cancelled) {
          setReadingLists([]);
          setSelectedListId(null);
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

  // console.log("Available languages:", manga.attributes.availableTranslatedLanguages);
  // console.log("Selected language:", selectedLanguage);
  // console.log("Filtered chapters:", filteredChapters);
  // console.log("Chapter data:", chapters[0]);

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

            {session?.access_token ? (
              <View style={styles.readingListBlock}>
                <Text style={styles.listPickerLabel}>Add to list</Text>
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
                          selectedListId === c.id && styles.listChipActive,
                        ]}
                        onPress={() => setSelectedListId(c.id)}
                      >
                        <Text
                          style={[
                            styles.listChipText,
                            selectedListId === c.id &&
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
                    (listBusy || selectedListId == null || listsLoading) &&
                      styles.addListBtnDisabled,
                  ]}
                  onPress={async () => {
                    if (!session.access_token || selectedListId == null) return;
                    setListMsg(null);
                    setListBusy(true);
                    try {
                      await addToReadingList(session.access_token, {
                        readingListId: selectedListId,
                        external_manga_id: manga.id,
                        manga_title: title,
                      });
                      setListMsg("Added to the selected list.");
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
                  disabled={listBusy || selectedListId == null || listsLoading}
                >
                  {listBusy ? (
                    <ActivityIndicator color="#fff" />
                  ) : (
                    <Text style={styles.addListBtnText}>
                      Add to selected list
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
            )}

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
                        router.push(`/reader/${chapter.id}`);
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
  readingListBlock: {
    marginBottom: 16,
  },
  listPickerLabel: {
    fontSize: 14,
    fontWeight: "600",
    color: "#444",
    marginBottom: 8,
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
