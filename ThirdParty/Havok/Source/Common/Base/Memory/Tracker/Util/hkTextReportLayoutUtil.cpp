/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/Util/hkTextReportLayoutUtil.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>


/* static */void HK_CALL hkTextReportLayoutUtil::calcColumnWidths(const char* texts, char delimiter, int numCols, const int* colWidths, int* colWidthsOut)
{
	hkLocalArray<hkSubString> colText(numCols);

	calcDelimitedText(texts, delimiter, numCols, colText);

	for (int i = 0; i < numCols; i++)
	{
		const hkSubString& text = colText[i];
		int width = text.length();

		if (colWidths)
		{
			int minWidth = colWidths[i];
			if (minWidth > 0)
			{
				width = hkMath::max2(minWidth, width);
			}
		}

		colWidthsOut[i] = width;
	}
}

/* static */void HK_CALL hkTextReportLayoutUtil::calcColumns(const int* colWidths, int numCols, int colSpacing, int startOffset, hkArray<Column>& columns)
{
	columns.setSize(numCols);

	for (int i = 0; i < numCols; i++)
	{
		if (i > 0)
		{
			startOffset += colSpacing;	
		}

		const int width = colWidths[i];

		Column& col = columns[i];
		col.m_start = startOffset;
		col.m_width = width;

		startOffset += width;
	}
}

/* static */void HK_CALL hkTextReportLayoutUtil::writeRepeatChar(char c, int size, hkOstream& stream)
{
	if (size > 0)
	{
		const int bufferSize = 16;
		char buffer[bufferSize + 1];

		{
			const int writeSize = hkMath::min2(size, bufferSize);
			for (int i = 0; i < writeSize; i++)
			{	
				buffer[i] = c;
			}
		}

		while (size > 0)
		{
			const int writeSize = hkMath::min2(size, bufferSize);
			stream.write(buffer, writeSize);
			size -= writeSize;
		}
	}
}

/* static */void HK_CALL hkTextReportLayoutUtil::writeAlignedText(Align align, const hkSubString& textIn, int colSize, hkOstream& stream)
{
	hkSubString text = textIn;

	// Make sure it fits
	{
		const int len = textIn.length();
		if (len > colSize)
		{
			text.m_end = text.m_start + colSize;
		}
	}
	const int len = text.length();
	const int space = colSize - len;

	switch (align)
	{
		case ALIGN_DEFAULT:
		case ALIGN_LEFT:
		{
			stream << text;
			if (space)
			{
				writeRepeatChar(' ', space, stream);
			}
			break;
		}
		case ALIGN_RIGHT:
		{
			if (space)
			{
				writeRepeatChar(' ', space, stream);
			}
			stream << text;
			break;
		}
		case ALIGN_CENTER:
		{
			int start = space / 2;
			int end = space - start;

			if (start)
			{
				writeRepeatChar(' ', start, stream);
			}
			stream << text;
			if (end)
			{
				writeRepeatChar(' ', end, stream);
			}
			break;
		}
	}
}

/* static */hkBool HK_CALL hkTextReportLayoutUtil::isValidAlignChar(char c)
{
	static const char validChars[] = "lrcd";
	for(const char* cur = validChars; *cur; cur++)
	{
		if (*cur == c)
		{
			return true;
		}
	}

	return false;
}

/* static */void HK_CALL hkTextReportLayoutUtil::writeColumns(const hkArray<Column>& columns, const char* alignChars, const hkSubString* texts, hkOstream& stream)
{
	Align defaultAlign = ALIGN_DEFAULT;
	int alignLen = 0;

	// Work out the default Align
	if (alignChars)
	{
		alignLen = hkString::strLen(alignChars);
		if (alignLen > 0 && isValidAlignChar(alignChars[alignLen - 1]))
		{
			defaultAlign = Align(alignChars[alignLen - 1]);
		}
	}

	int pos = 0;
	for (int i = 0; i < columns.getSize(); i++)
	{
		const Column& col = columns[i];
		if (col.m_start > pos)
		{
			writeRepeatChar(' ', col.m_start - pos, stream);
			pos = col.m_start;
		}

		Align align = defaultAlign;
		if (i < alignLen)
		{
			char alignChar = alignChars[i];
			HK_ASSERT(0x454a532a, isValidAlignChar(alignChar));
			if (isValidAlignChar(alignChar))
			{
				align = Align(alignChar);
			}
		}
		
		// Get the text
		writeAlignedText(align, texts[i], col.m_width, stream);

		pos += col.m_width;
	}
}

/* static */void HK_CALL hkTextReportLayoutUtil::writeColumns( const hkArray<Column>& columns, const char* alignChars, const char*const* texts, hkOstream& stream)
{
	hkLocalBuffer<hkSubString> subStrings(columns.getSize());

	for (int i = 0; i < columns.getSize(); i++)
	{
		subStrings[i] = texts[i];
	}

	writeColumns(columns, alignChars, subStrings.begin(), stream);
}

/* static */void HK_CALL hkTextReportLayoutUtil::writeColumns(const hkArray<Column>& columns, const char* alignChars, const char* texts, char delimiter, hkOstream& stream)
{
	hkLocalArray<hkSubString> subStrings(columns.getSize());

	hkTextReportLayoutUtil::calcDelimitedText(texts, delimiter, columns.getSize(), subStrings);

	writeColumns(columns, alignChars, subStrings.begin(), stream);
}

/* static */void HK_CALL hkTextReportLayoutUtil::calcDelimitedText(const char* texts, char delimiter, hkArray<hkSubString>& subStrings)
{
	subStrings.clear();

	// Handle the empty case
	if (*texts == 0)
	{
		hkSubString& subString = subStrings.expandOne();
		subString.set(texts, texts);
		return;
	}

	const char* start = texts;
	while (*start)
	{
		const char* end = start;
		while (*end != 0 && *end != delimiter) end++;

		subStrings.expandOne().set(start, end);

		start = end;
		if (*start == delimiter)
		{
			start++;
		}
	}
}

/* static */void HK_CALL hkTextReportLayoutUtil::calcDelimitedText(const char* texts, char delimiter, int numCols, hkArray<hkSubString>& subStrings)
{
	calcDelimitedText(texts, delimiter, subStrings);

	if (subStrings.getSize() > numCols)
	{
		// Remove extras
		subStrings.setSize(numCols);
	}
	else if (subStrings.getSize() < numCols)
	{
		// Fill in the end with blanks
		hkSubString back = subStrings.back();
		while (subStrings.getSize() < numCols)
		{
			subStrings.expandOne().set(back.m_end, back.m_end);
		}
	}
}

/* static */ void HK_CALL hkTextReportLayoutUtil::writeCharColumns(const hkArray<Column>& columns, char c, hkOstream& stream)
{
	int pos = 0;
	for (int i = 0; i < columns.getSize(); i++)
	{
		const Column& col = columns[i];
		if (col.m_start > pos)
		{
			writeRepeatChar(' ', col.m_start - pos, stream);
			pos = col.m_start;
		}

		writeRepeatChar(c, col.m_width, stream);

		pos += col.m_width;
	}
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
