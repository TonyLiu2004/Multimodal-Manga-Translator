import React, { useEffect, useRef, useState } from 'react';
import { View, FlatList, Image, StyleSheet, Text, useWindowDimensions, TouchableOpacity, Pressable, ActivityIndicator, Switch } from 'react-native';
import { BACKEND_URL } from "@/config";

interface MangaReaderProps {
  pages: string[]; // Array of image URLs for manga pages
}

interface TranslationBubble {
    bubble_index: number;
    original_text: string;
    translated_text: string;
}

export let currentMangaPage = 1;

export default function MangaReader({ pages }: MangaReaderProps) {
    const { width: screenWidth, height: screenHeight } = useWindowDimensions();
    const [showMenu, setShowMenu] = useState(true);
    const [currentPage, setCurrentPage] = useState(1);
    const [autoTranslateEnabled, setAutoTranslateEnabled] = useState(true);
    const [translationsByPage, setTranslationsByPage] = useState<Record<number, TranslationBubble[]>>({});
    const [inFlightByPage, setInFlightByPage] = useState<Record<number, boolean>>({});
    const inFlightPagesRef = useRef<Set<number>>(new Set());
    const abortControllersRef = useRef<Map<number, AbortController>>(new Map());
    const LOOKAHEAD_PANELS = 3;
    const KEEP_PREVIOUS_PAGES = 1;
    const SCROLL_EVENT_THROTTLE = 100;
    // Fallback
    if (pages.length === 0) {
        return (
            <View style={[styles.pageContainer, styles.centered]}>
                <Text>No pages available</Text>
            </View>
        );
    }

    const renderPage = ({ item, index }: { item: string, index: number }) => (
        <View style={[styles.pageContainer, { width: screenWidth, height: screenHeight }]}>
            <Image
                source={{ uri: item }}
                style={styles.page}
                resizeMode="contain"
            />
            <Text style={styles.pageNumber}>Page {index + 1}/{pages.length}</Text>
        </View>
    )

    const isValidPage = (pageNumber: number) => pageNumber >= 1 && pageNumber <= pages.length;

    const markPageInFlight = (pageNumber: number, inFlight: boolean) => {
        setInFlightByPage((prev) => ({
            ...prev,
            [pageNumber]: inFlight,
        }));
    };

    const getPageFromScroll = (offsetY: number, viewportHeight: number) => {
        const rawIndex = Math.round(offsetY / viewportHeight);
        return Math.max(0, Math.min(rawIndex, pages.length - 1)) + 1;
    };

    const requestPageTranslation = async (pageNumber: number) => {
        if (!isValidPage(pageNumber)) {
            return;
        }

        //Page already translated and in cache or is currently being processed
        if (translationsByPage[pageNumber] || inFlightPagesRef.current.has(pageNumber)) {
            console.log(`skipping page ${pageNumber}`)
            return;
        }
        console.log(`Translate page ${pageNumber}`)

        inFlightPagesRef.current.add(pageNumber);
        markPageInFlight(pageNumber, true);
        const abortController = new AbortController();
        abortControllersRef.current.set(pageNumber, abortController);

        const backendUrl = `${BACKEND_URL}/api/manga/translate`
        try {
            const res = await fetch(backendUrl, {
                method: 'POST',
                body: JSON.stringify({
                    image_url: pages[pageNumber - 1],
                    language: '',
                }),
                headers: {
                    'Content-Type': 'application/json',
                },
                signal: abortController.signal,
            });
            const json = await res.json();

            if (json?.status === 'success' && Array.isArray(json?.data)) {
                setTranslationsByPage((prev) => ({
                    ...prev,
                    [pageNumber]: json.data as TranslationBubble[],
                }));
            }
        } catch (err) {
            if (err instanceof Error && err.name === 'AbortError') {
                console.log(`Cancelled translation for page ${pageNumber}`);
            } else {
                console.error(`Failed to translate page ${pageNumber}`, err);
            }
        } finally {
            inFlightPagesRef.current.delete(pageNumber);
            abortControllersRef.current.delete(pageNumber);
            markPageInFlight(pageNumber, false);
        }
    }

    const handleTranslate = async () => {
        await requestPageTranslation(currentPage);
    }

    const prefetchLookaheadPages = (basePage: number) => {
        for (let i = 0; i <= LOOKAHEAD_PANELS; i++) {
            const pageToTranslate = basePage + i;
            if (pageToTranslate <= pages.length) {
                void requestPageTranslation(pageToTranslate);
            }
        }
    };

    const cancelStaleInFlightRequests = (basePage: number) => {
        // Keep only nearby in-flight translation requests
        const minPageToKeep = Math.max(1, basePage - KEEP_PREVIOUS_PAGES);
        const maxPageToKeep = Math.min(pages.length, basePage + LOOKAHEAD_PANELS);

        for (const page of inFlightPagesRef.current) {
            const outsideWindow = page < minPageToKeep || page > maxPageToKeep;
            if (outsideWindow) {
                abortControllersRef.current.get(page)?.abort();
            }
        }
    };

    useEffect(() => {
        if (!autoTranslateEnabled) {
            return;
        }
        cancelStaleInFlightRequests(currentPage);
        prefetchLookaheadPages(currentPage);
    }, [currentPage, autoTranslateEnabled]);

    const handleScroll = (offsetY: number, viewportHeight: number) => {
        const nextPage = getPageFromScroll(offsetY, viewportHeight);
        currentMangaPage = nextPage;

        if (nextPage !== currentPage) {
            setCurrentPage(nextPage);
        }
        if (showMenu) {
            setShowMenu(false);
        }
    };

    return (
        <View style={{ flex: 1 }}>
            {showMenu && (
                <View style={styles.menuContainer}>
                    <TouchableOpacity style={styles.menuButton} onPress={() => handleTranslate()}>
                        <Text style={styles.menuButtonText}>Translate</Text>
                    </TouchableOpacity>
                    <View style={styles.autoTranslateToggleRow}>
                        <Text style={styles.autoTranslateToggleText}>Auto translate</Text>
                        <Switch
                            value={autoTranslateEnabled}
                            onValueChange={setAutoTranslateEnabled}
                            thumbColor="#fff"
                            trackColor={{ false: 'rgba(255,255,255,0.25)', true: 'rgba(34,197,94,0.7)' }}
                        />
                    </View>
                    {inFlightByPage[currentPage] && (
                        <View style={styles.inFlightIndicator}>
                            <ActivityIndicator size="small" color="#fff" />
                            <Text style={styles.inFlightText}>Translating current page...</Text>
                        </View>
                    )}

                    {translationsByPage[currentPage] && (
                        <View style={styles.translationPanel}>
                            <Text style={styles.translationTitle}>Page {currentPage} translations</Text>
                            {translationsByPage[currentPage].map((bubble) => (
                                <View key={bubble.bubble_index} style={styles.translationItem}>
                                    <Text style={styles.translationText}>{bubble.original_text}</Text>
                                    <Text style={styles.translationText}>{bubble.translated_text}</Text>
                                </View>
                            ))}
                        </View>
                    )}
                </View>
            )}
            
            <Pressable style={{ flex: 1 }} onPress={() => setShowMenu(true)}>
                <FlatList
                    data={pages}
                    keyExtractor={(item, index) => index.toString()}
                    renderItem={renderPage}
                    initialNumToRender={3}
                    maxToRenderPerBatch={10}
                    windowSize={100}
                    onScroll={(e) => {
                        const offsetY = e.nativeEvent.contentOffset.y;
                        const viewportHeight = e.nativeEvent.layoutMeasurement.height || screenHeight;
                        handleScroll(offsetY, viewportHeight);
                    }}
                    scrollEventThrottle={SCROLL_EVENT_THROTTLE}
                />
            </Pressable>
        </View>
    );
}

const styles = StyleSheet.create({
    pageContainer: {
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#000',
    },
    page: {
        width: '100%',
        height: '100%',
    },
    pageNumber: {
        position: 'absolute',
        bottom: 20,
        alignSelf: 'center',
        color: '#fff',
        fontSize: 14,
        fontWeight: '600',
        backgroundColor: 'rgba(0, 0, 0, 0.4)',
        paddingHorizontal: 10,
        paddingVertical: 4,
        borderRadius: 8,
        overflow: 'hidden',
        zIndex: 10,
    },
    menuContainer: {
        position: 'absolute',
        top: 16,
        alignSelf: 'center',
        zIndex: 20,
        alignItems: 'center',
        width: '92%',
    },
    menuButton: {
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        paddingHorizontal: 14,
        paddingVertical: 8,
        borderRadius: 10,
    },
    menuButtonText: {
        color: '#fff',
        fontSize: 14,
        fontWeight: '600',
    },
    autoTranslateToggleRow: {
        marginTop: 8,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        width: '100%',
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        borderRadius: 10,
        paddingHorizontal: 12,
        paddingVertical: 6,
    },
    autoTranslateToggleText: {
        color: '#fff',
        fontSize: 12,
        fontWeight: '500',
    },
    inFlightIndicator: {
        marginTop: 8,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        borderRadius: 10,
        paddingHorizontal: 12,
        paddingVertical: 6,
    },
    inFlightText: {
        color: '#fff',
        fontSize: 12,
        fontWeight: '500',
    },
    translationPanel: {
        marginTop: 10,
        width: '100%',
        maxHeight: 250,
        backgroundColor: 'rgba(0, 0, 0, 0.75)',
        borderRadius: 10,
        padding: 10,
    },
    translationTitle: {
        color: '#fff',
        fontSize: 13,
        fontWeight: '700',
        marginBottom: 8,
    },
    translationItem: {
        marginBottom: 8,
        paddingBottom: 8,
        borderBottomWidth: 1,
        borderBottomColor: 'rgba(255, 255, 255, 0.2)',
    },
    translationText: {
        color: '#fff',
        fontSize: 12,
        lineHeight: 18,
    },
    centered: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
});