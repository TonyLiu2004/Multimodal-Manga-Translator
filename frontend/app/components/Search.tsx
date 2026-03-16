import { useWindowDimensions, Text, ScrollView, View, StyleSheet, ActivityIndicator, TextInput, Image, Pressable, FlatList, NativeSyntheticEvent, NativeScrollEvent } from "react-native";
import React, { useEffect, useState, useRef } from 'react';

interface Chapter {
  id: string;
  chapter: string;
  title: string;
  pages: number;
}

interface SearchProps {
    query: string;
}

export default function Search({query} : SearchProps) {
    const [mangaList, setMangaList] = useState<any[]>([]);
    const [popupVisible, setPopupVisible] = useState(false);
    const [selectedManga, setSelectedManga] = useState<{ title: string; summary: string; coverUrl: string; mangaId: string }>({ title: '', summary: '', coverUrl: '', mangaId: '' });
    const [chapters, setChapters] = useState<Chapter[]>([]);
    const [loadingChapters, setLoadingChapters] = useState(false);

    const { width } = useWindowDimensions();
    const isDesktop = width > 600;

    const handleSearch = async () => {
    console.log("Searching for:", query);
    if (!query) return;
    try {
        const response = await fetch(`https://api.mangadex.org/manga?title=${query}&limit=10&includes[]=cover_art`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
        });
        
        if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const json = await response.json();
        setMangaList(json.data || []);
        console.log("Search results:", json.data);
    } catch (error) {
        console.error("Search error:", error);
        setMangaList([]);
    }
    };

    const fetchChapters = async (mangaId: string) => {
        setLoadingChapters(true);
        try {
        const response = await fetch(
            `https://api.mangadex.org/manga/${mangaId}/feed?limit=20&order[chapter]=asc&translatedLanguage[]=en`
        );
        const json = await response.json();
        const chapterData: Chapter[] = (json.data || []).map((ch: any) => ({
            id: ch.id,
            chapter: ch.attributes.chapter || '?',
            title: ch.attributes.title || '',
            pages: ch.attributes.pages || 0,
        }));
        setChapters(chapterData);
        console.log(`Fetched ${chapterData.length} chapters`);
        } catch (error) {
        console.error("Error fetching chapters:", error);
        setChapters([]);
        } finally {
        setLoadingChapters(false);
        }
    };

    return (
        <ScrollView>
            {mangaList.length > 0 && (
                <View style={{ marginTop: 20, alignItems: 'center' }}>
                <Text>{mangaList.length} manga found!</Text>
                <View style={{ marginTop: 20, alignItems: 'center' }}>
                    {mangaList.map((manga) => {
                    const titles = Object.values(manga.attributes.title);
                    const displayTitle = titles[0] as string || "Untitled";

                    // get cover art url
                    const coverArt = manga.relationships.find((rel: any) => rel.type === 'cover_art');
                    const fileName = coverArt?.attributes?.fileName;
                    const coverUrl = fileName 
                        ? `https://uploads.mangadex.org/covers/${manga.id}/${fileName}.256.jpg`
                        : 'https://via.placeholder.com/256x360?text=No+Cover';
                    return (
                        <Pressable 
                        key={manga.id} 
                        style={{ 
                            marginBottom: 10, 
                            justifyContent: 'flex-start',
                            flexDirection: isDesktop ? 'row' : 'column',
                            width: '50%',
                        }}
                        onPress={() => {
                            setSelectedManga({
                            title: displayTitle,
                            summary: manga.attributes.description.en || "No description available.",
                            coverUrl: coverUrl,
                            mangaId: manga.id
                            });
                            setPopupVisible(true);
                            fetchChapters(manga.id);
                        }}
                        >
                        <Image 
                            source={{ uri: coverUrl }} 
                            style={{ width: 256, height: 360 }}
                        />
                        <View style={{ flex: 1, marginLeft: 30 }}>
                            <Text style={{ fontWeight: 'bold' }}>{displayTitle}</Text>
                            <Text style={{ color: 'gray' }}>ID: {manga.id}</Text>
                            <Text>{manga.attributes.description.en || "No description available."}</Text>
                        </View>
                        </Pressable>
                    );
                    })}
                </View>
                </View>
            )}
      </ScrollView>
    )
}