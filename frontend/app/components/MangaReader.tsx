import React, { useState } from 'react';
import { View, FlatList, Image, StyleSheet, Text, useWindowDimensions, TouchableOpacity, Pressable } from 'react-native';
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
    const [translationsByPage, setTranslationsByPage] = useState<Record<number, TranslationBubble[]>>({});
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

    const handleTranslate = async () => {
        console.log(`handleTranslate: page ${currentPage}/${pages.length}`);
        console.log(pages[currentPage - 1]);
        const backendUrl = `${BACKEND_URL}/api/manga/translate`
        const res = await fetch(backendUrl, {
            method: 'POST',
            body: JSON.stringify({
                image_url: pages[currentPage - 1],
                language: '',
            }),
            headers: {
                'Content-Type': 'application/json',
            },
        });
        const json = await res.json();
        console.log(json);

        if (json?.status === 'success' && Array.isArray(json?.data)) {
            setTranslationsByPage((prev) => ({
                ...prev,
                [currentPage]: json.data as TranslationBubble[],
            }));
        }

        return json;
    }

    return (
        <View style={{ flex: 1 }}>
            {showMenu && (
                <View style={styles.menuContainer}>
                    <TouchableOpacity style={styles.menuButton} onPress={() => handleTranslate()}>
                        <Text style={styles.menuButtonText}>Translate</Text>
                    </TouchableOpacity>

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
                        const index = Math.round(offsetY / viewportHeight);
                        const nextPage = Math.max(0, Math.min(index, pages.length - 1)) + 1;
                        currentMangaPage = nextPage;
                        if (nextPage !== currentPage) {
                            setCurrentPage(nextPage);
                        }
                        if (showMenu) {
                            setShowMenu(false);
                        }
                        // console.log(`Page ${nextPage}/${pages.length}`);
                    }}
                    scrollEventThrottle={100}
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