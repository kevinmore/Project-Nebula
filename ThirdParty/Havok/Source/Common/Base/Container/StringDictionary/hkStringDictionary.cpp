/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Container/StringDictionary/hkStringDictionary.h>

hkStringDictionary::hkStringDictionary( const char** strings, int numStrings )
{
	m_dictionary.append( strings, numStrings );
}

hkStringDictionary::~hkStringDictionary()
{
}

void hkStringDictionary::insert( const char* word )
{
	HK_ASSERT2(0x306ba14, !containsWord(word), "Word is already in the dictionary.");
	m_dictionary.pushBack( word );
}

bool hkStringDictionary::remove( const char* word )
{
	const int numWords = m_dictionary.getSize();
	for (int wordIndex = 0; wordIndex < numWords; ++wordIndex)
	{
		if (hkString::strCmp( m_dictionary[wordIndex], word ) == 0)
		{
			m_dictionary.removeAt(wordIndex);
			return true;
		}
	}

	return false;
}

void hkStringDictionary::clear()
{
	m_dictionary.clear();
}

bool hkStringDictionary::containsWord( const char* word ) const
{
	const int numWords = m_dictionary.getSize();
	for (int wordIndex = 0; wordIndex < numWords; ++wordIndex)
	{
		if (hkString::strCmp( m_dictionary[wordIndex], word ) == 0)
		{			
			return true;
		}
	}

	return false;
}

void hkStringDictionary::findMatches( const char* pattern, hkArray<const char*>& resultsOut, hkMemoryAllocator& resultsAllocator, const char wildcard, const bool caseSensitive ) const
{	
	int dictionarySize = m_dictionary.getSize();
	int searchResult = -1;
	for (int stringIndex = 0; stringIndex < dictionarySize; ++stringIndex)
	{
		searchResult = naiveStringSearch( m_dictionary[stringIndex], pattern, wildcard );
		if (searchResult > -1)
		{
			resultsOut._pushBack( resultsAllocator, m_dictionary[stringIndex] );
		}
	}
}

void hkStringDictionary::copyWordsFromDictionary( hkArray<const char*>& wordsOut ) const
{
	int size = m_dictionary.getSize();
	wordsOut.reserveExactly( size );
	for (int i = 0; i < size; ++i)
	{
		wordsOut.pushBack( m_dictionary[i] );
	}
}

int hkStringDictionary::getSize() const
{
	return m_dictionary.getSize();
}

int hkStringDictionary::naiveStringSearch( const char* text, const char* pattern, const char wildcard, const bool caseSensitive ) const
{
	// Search for matches using a naive per-character comparison search.
	// The given wildcard character represent 0-or-more non-matching characters.
	// For example, a pattern of "c r n" with wildcard of ' ' would match "corn" and "crying".
	int indexInPattern;

	for (int indexInText = 0; text[indexInText] != '\0'; ++indexInText)
	{
		for (indexInPattern = 0; 
			 ((text[indexInText + indexInPattern] != '\0') && (pattern[indexInPattern] != '\0')); 
			 ++indexInPattern )
		{
			if (wildcard != '\0' && pattern[indexInPattern] == wildcard)
			{
				int offsetDueToWildcard = 0;
				// Consume all wildcard characters
				while (pattern[indexInPattern] == wildcard)
				{
					++indexInPattern;
					--offsetDueToWildcard;
				}
				if (pattern[indexInPattern] == '\0')
				{
					// The end of the string, that had matched up to this point, is all wildcards.
					// So this is a match.
					return indexInText;
				}

				// Find the suffix of the text that matches the first post-wildcard character in the pattern,
				// and search for the pattern suffix in this new text.
				while ( !charCmpFollowingCaseRules(text[indexInText + indexInPattern + offsetDueToWildcard], 
												   pattern[indexInPattern],
												   caseSensitive) )
				{
					if (text[indexInText + indexInPattern + offsetDueToWildcard] == '\0')
					{
						return -1;
					}

					offsetDueToWildcard++;
				}
				if (-1 < naiveStringSearch( &(text[indexInText + indexInPattern + offsetDueToWildcard]) , &pattern[indexInPattern], wildcard, caseSensitive ) )
				{
					// The wildcard suffix matched. Return the index in the text of the first matching character.
					return (indexInText + indexInPattern + offsetDueToWildcard);
				}
				else
				{
					return -1;
				}
			}
			else if ( charCmpFollowingCaseRules( text[indexInText + indexInPattern], pattern[indexInPattern], caseSensitive ) )
			{
				// Characters match following the dictionary's case sensitivity rules; continue comparison loop.
			}
			else
			{
				// Non-wildcard mismatch. Exit comparison loop.
				break;
			}
		}

		// Consume all trailing wildcards to handle cases where a pattern can be longer than the text due to
		// trailing wildcards. Without doing this, pattern "bat " would return a false negative for text "bat".
		if ( wildcard != '\0' )
		{
			while (pattern[indexInPattern] == wildcard)
			{
				++indexInPattern;
			}
		}

		// If the pattern was fully matched, return the index in the text at which the match began.
		if (pattern[indexInPattern] == '\0')
		{
			// Found a match
			return indexInText;
		}
	}

	return -1;
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
