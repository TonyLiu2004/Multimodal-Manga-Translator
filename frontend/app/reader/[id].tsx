import { useLocalSearchParams } from 'expo-router';
import { Asset } from 'expo-asset';
import { useEffect, useState } from 'react';
import { ActivityIndicator, View } from 'react-native';
import MangaReader from '../components/MangaReader'; 

import { BACKEND_URL } from "@/config";

const IS_TESTING = false; // true for testing with local images

export default function ReaderScreen() {
    const { id } = useLocalSearchParams();
    const [pages, setPages] = useState<string[]>([]);
    const [loading, setLoading] = useState(true);

    const runTest = async () => {
        const testPages = [
            Asset.fromModule(require('../../assets/images/test_1.png')).uri,
            Asset.fromModule(require('../../assets/images/test_7.png')).uri,
            Asset.fromModule(require('../../assets/images/cntest_1.png')).uri,
            Asset.fromModule(require('../../assets/images/krtest_1.png')).uri,
        ];

        setPages(testPages);
    };

    // Placeholder
    const runBackend = async () => {
        try {
            const res = await fetch(`${BACKEND_URL}/api/manga/chapter/${id}/pages`);

            if (!res.ok) {
                throw new Error(`Server responded with ${res.status}`);
            }

            const json = await res.json();
            if (json.urls && Array.isArray(json.urls)) {
                setPages(json.urls);
            } else {
                console.warn("Backend returned no URLs for this chapter.");
                setPages([]);
            }
        } catch (error) {
            console.error("Error fetching chapter pages:", error);
            setPages([]);
        }
    }

    useEffect(() => {
        const fetchPages = async () => {
            setLoading(true);
            try {
                if (IS_TESTING) {
                    await runTest();
                } else {
                    await runBackend();
                }
            } catch (error) {
                console.error("Failed to fetch chapter pages:", error);
            } finally {
                setLoading(false);
            }
        };

        if (id) {
            fetchPages();
        }
    }, [id]);

    if (loading) {
        return (
            <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
                <ActivityIndicator size="large" color="#007AFF" />
            </View>
        );
    }

    return <MangaReader pages={pages} />; 
}