import { useRouter } from "expo-router";
import { Modal, Text, ScrollView, View, StyleSheet, Pressable, Image, ActivityIndicator } from "react-native";
import React from 'react';
import { Chapter } from "../types/types";
import { Manga } from "../types/types";

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

export default function PopUp({ visible, title, summary, coverArt, chapters, loadingChapters, manga, onClose }: PopUpProps) {

    const router = useRouter();

    const [selectedLanguage, setSelectedLanguage] = React.useState<string>('All');
    const filteredChapters = chapters.filter(ch => 
        selectedLanguage === 'All' || ch.language === selectedLanguage
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
            <Pressable 
                style={styles.overlay}
                onPress={onClose}
            >
                <Pressable 
                    style={styles.popup}
                    onPress={(e) => e.stopPropagation()}
                >
                    <ScrollView style={styles.content}>
                        <Image 
                            source={{ uri: coverArt }} 
                            style={styles.coverImage}
                        />
                        <Text style={styles.title}>{title}</Text>
                        <Text style={styles.summary}>{summary}</Text>
                        
                        <View style={styles.chaptersSection}>
                            <Text style={styles.chaptersTitle}>Chapters</Text>
                            <Text style={styles.languageLabel}>Select Language:</Text>
                            <View style={styles.languageContainer}>
                                {['All', ...manga.attributes.availableTranslatedLanguages].map((lang) => (
                                    <Pressable
                                        key={lang}
                                        style={[
                                            styles.langChip,
                                            selectedLanguage === lang && styles.activeLangChip
                                        ]}
                                        onPress={() => setSelectedLanguage(lang)}
                                    >
                                        <Text style={[
                                            styles.langChipText,
                                            selectedLanguage === lang && styles.activeLangChipText
                                        ]}>
                                            {lang.toUpperCase()}
                                        </Text>
                                    </Pressable>
                                ))}
                            </View>
                            {loadingChapters ? (
                                <ActivityIndicator size="large" color="#007AFF" style={{ marginTop: 20 }} />
                            ) : filteredChapters.length > 0 ? (
                                <View style={styles.chaptersList}>
                                    {filteredChapters.map((chapter) => (
                                        <Pressable key={chapter.id} style={styles.chapterItem} onPress={() => {
                                            onClose();
                                            router.push(`/reader/${chapter.id}`);
                                        }}>
                                            <Text style={styles.chapterNumber}>Ch. {chapter.chapter}</Text>
                                            <Text style={styles.chapterTitle}>
                                                {chapter.title || 'No title'}
                                            </Text>
                                            <Text style={styles.chapterPages}>{chapter.pages} pages</Text>
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
    )
}

const styles = StyleSheet.create({
    overlay: {
        flex: 1,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    popup: {
        backgroundColor: 'white',
        borderRadius: 15,
        padding: 20,
        width: '80%',
        maxWidth: 600,
        maxHeight: '80%',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.25,
        shadowRadius: 4,
        elevation: 5,
    },
    content: {
        marginBottom: 15,
    },
    coverImage: {
        width: '100%',
        height: 300,
        borderRadius: 10,
        marginBottom: 15,
        resizeMode: 'contain',
    },
    title: {
        fontSize: 24,
        fontWeight: 'bold',
        marginBottom: 15,
        color: '#333',
    },
    summary: {
        fontSize: 16,
        lineHeight: 24,
        color: '#666',
        marginBottom: 20,
    },
    chaptersSection: {
        marginTop: 20,
        paddingTop: 20,
        borderTopWidth: 1,
        borderTopColor: '#E0E0E0',
    },
    chaptersTitle: {
        fontSize: 20,
        fontWeight: 'bold',
        marginBottom: 15,
        color: '#333',
    },
    chaptersList: {
        gap: 10,
    },
    chapterItem: {
        padding: 12,
        backgroundColor: '#F5F5F5',
        borderRadius: 8,
        marginBottom: 8,
    },
    chapterNumber: {
        fontSize: 16,
        fontWeight: '600',
        color: '#007AFF',
        marginBottom: 4,
    },
    chapterTitle: {
        fontSize: 14,
        color: '#333',
        marginBottom: 4,
    },
    chapterPages: {
        fontSize: 12,
        color: '#999',
    },
    noChapters: {
        fontSize: 14,
        color: '#999',
        fontStyle: 'italic',
        textAlign: 'center',
        marginTop: 10,
    },
    languageLabel: {
        fontSize: 14,
        fontWeight: '600',
        color: '#666',
        marginBottom: 10,
    },
    languageContainer: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 8,
        marginBottom: 20,
    },
    langChip: {
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 20,
        backgroundColor: '#F0F0F0',
        borderWidth: 1,
        borderColor: '#DDD',
    },
    activeLangChip: {
        backgroundColor: '#007AFF',
        borderColor: '#007AFF',
    },
    langChipText: {
        fontSize: 12,
        color: '#333',
        fontWeight: '500',
    },
    activeLangChipText: {
        color: '#FFF',
        fontWeight: 'bold',
    },
    closeButton: {
        backgroundColor: '#007AFF',
        padding: 12,
        borderRadius: 8,
        alignItems: 'center',
    },
    closeButtonText: {
        color: 'white',
        fontSize: 16,
        fontWeight: '600',
    },
});