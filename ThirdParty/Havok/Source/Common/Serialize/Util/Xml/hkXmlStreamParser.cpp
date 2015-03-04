/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Util/Xml/hkXmlStreamParser.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Serialize/Util/Xml/hkFloatParseUtil.h>

hkXmlStreamParser::hkXmlStreamParser(hkStreamReader* reader):
	m_lex(reader) 
{
	m_token = TOKEN_ERROR;
}

hkXmlStreamParser::Token hkXmlStreamParser::_parseBlock()
{
	hkXmlLexAnalyzer::Token tok = m_lex.advance();
	if (tok == hkXmlLexAnalyzer::TOKEN_SLASH)
	{
		// Looks like an end block
		tok = m_lex.advance();

		if (tok != hkXmlLexAnalyzer::TOKEN_IDENTIFIER)
		{
			return TOKEN_ERROR;
		}
		_pushLexeme();
		tok = m_lex.advance();
		if (tok == hkXmlLexAnalyzer::TOKEN_END_BLOCK)
		{
			return TOKEN_BLOCK_END;
		}
		return TOKEN_ERROR;
	}

	if (tok != hkXmlLexAnalyzer::TOKEN_IDENTIFIER)
	{
		return TOKEN_ERROR;
	}
	_pushLexeme();

	while (true)
	{
		tok = m_lex.advance();

		switch (tok)
		{
			case hkXmlLexAnalyzer::TOKEN_IDENTIFIER:
			{
				break;
			}
			case hkXmlLexAnalyzer::TOKEN_SLASH:
			{
				tok = m_lex.advance();

				_extractAttributes();

				return (tok == hkXmlLexAnalyzer::TOKEN_END_BLOCK) ? TOKEN_BLOCK_START_END : TOKEN_ERROR;
			}
			case hkXmlLexAnalyzer::TOKEN_END_BLOCK:
			{
				_extractAttributes();
				return TOKEN_BLOCK_START;
			}
			default:
			{
				return TOKEN_ERROR;
			}
		}	

		// Save the value name
		_pushLexeme();

		tok = m_lex.advance();
		if (tok != hkXmlLexAnalyzer::TOKEN_EQUALS)
		{
			return TOKEN_ERROR;
		}

		tok = m_lex.advance();
		if (tok != hkXmlLexAnalyzer::TOKEN_QUOTED_STRING)
		{
			return TOKEN_ERROR;
		}

		// Save the value
		_pushLexeme();
	}
}

hkXmlStreamParser::Token hkXmlStreamParser::_parseQBlock()
{
	hkXmlLexAnalyzer::Token tok = m_lex.advance();
	if (tok != hkXmlLexAnalyzer::TOKEN_IDENTIFIER)
	{
		return TOKEN_ERROR;
	}

	_pushLexeme();
	while (true)
	{
		tok = m_lex.advance();

		switch (tok)
		{
			case hkXmlLexAnalyzer::TOKEN_IDENTIFIER:
			{
				break;
			}
			case hkXmlLexAnalyzer::TOKEN_END_QBLOCK:
			{
				_extractAttributes();
				return TOKEN_QBLOCK;
			}
			default:
			{
				return TOKEN_ERROR;
			}
		}	

		// Save the value name
		_pushLexeme();
		tok = m_lex.advance();
		if (tok != hkXmlLexAnalyzer::TOKEN_EQUALS)
		{
			return TOKEN_ERROR;
		}

		tok = m_lex.advance();
		if (tok != hkXmlLexAnalyzer::TOKEN_QUOTED_STRING)
		{
			return TOKEN_ERROR;
		}
		// Save the value
		_pushLexeme();
	}
}

void hkXmlStreamParser::_pushLexeme()
{
	SubString& subString = m_lexemes.expandOne();
	subString.m_start = m_lex.getLexemeStartIndex();
	subString.m_end = m_lex.getLexemeEndIndex();
}

void hkXmlStreamParser::_extractAttributes()
{
	m_attribMap.clear();
	m_keys.clear();

	int totalSize = 0;
	for (int i = 1; i < m_lexemes.getSize(); i += 2)
	{
		const SubString& key = m_lexemes[i];
		const int len = (key.m_end - key.m_start);
		HK_ASSERT(0x254534a5, len > 0);

		totalSize += len + 1;
	}

	m_keyStorage.setSize(totalSize);
	char* keysOut = m_keyStorage.begin();

	for (int i = 1; i < m_lexemes.getSize(); i += 2)
	{
		const SubString& key = m_lexemes[i];
		const int len = (key.m_end - key.m_start);
		hkSubString keyString = m_lex.getLexeme(key.m_start, key.m_end);

		hkString::strNcpy(keysOut, keyString.m_start, len);
		keysOut[len] = 0;

		m_keys.pushBack(keysOut);

		m_attribMap.insert(keysOut, i + 1);
		keysOut += len + 1;
	}
}

hkXmlStreamParser::Token hkXmlStreamParser::_advance()
{
	m_keys.clear();
	m_attribMap.clear();
	m_lex.bufferCommit();
	m_lexemes.clear();

	while (true)
	{
		hkXmlLexAnalyzer::Token tok = m_lex.advance();
		switch (tok)
		{
			case hkXmlLexAnalyzer::TOKEN_START_BLOCK:
			{
				return _parseBlock();
			}
			case hkXmlLexAnalyzer::TOKEN_START_QBLOCK:
			{
				return _parseQBlock();
			}
			case hkXmlLexAnalyzer::TOKEN_TEXT:
			{
				return TOKEN_TEXT;
			}
			case hkXmlLexAnalyzer::TOKEN_WHITESPACE:
			{
				return TOKEN_WHITESPACE;
			}
			case hkXmlLexAnalyzer::TOKEN_COMMENT:
			{
				// Ignore
				break;
			}
			case hkXmlLexAnalyzer::TOKEN_EOF:
			{
				return TOKEN_EOF;
			}
			default:
			case hkXmlLexAnalyzer::TOKEN_ERROR:
			{
				return TOKEN_ERROR;
			}
		}
	}
}

hkXmlStreamParser::Token hkXmlStreamParser::advance()
{
	return m_token = _advance();
}

hkResult hkXmlStreamParser::getValue(const char* key, hkSubString& valueOut) const
{
	int index = m_attribMap.getWithDefault(key, -1);
	if (index < 0)
	{
		return HK_FAILURE;
	}

	const SubString& value = m_lexemes[index];
	valueOut = m_lex.getLexeme(value.m_start, value.m_end);

	return HK_SUCCESS;
}

hkSubString hkXmlStreamParser::getBlockName() const
{
	if (m_lexemes.getSize() > 0)
	{
		const SubString& name = m_lexemes[0];
		return m_lex.getLexeme(name.m_start, name.m_end);
	}
	else
	{
		return hkSubString(HK_NULL, HK_NULL);
	}
}

hkSubString hkXmlStreamParser::getLexeme() const
{
	return m_lex.getLexeme();
}


static void _dumpAttributes(hkXmlStreamParser& parser, hkOstream& stream)
{
	stream << parser.getBlockName();

	const hkArray<const char*>& keys = parser.getKeys();

	for (int i = 0; i < keys.getSize(); i++)
	{
		const char* key = keys[i];

		stream << " " << key << "=";
		hkSubString value;
		parser.getValue(key, value);
		stream << value;
	}
}

void hkXmlStreamParser::dumpParse(hkOstream& out)
{
	while (true)
	{
		hkXmlStreamParser::Token tok = advance();

		switch (tok)
		{
			case hkXmlStreamParser::TOKEN_QBLOCK:
			{
				out << "<?";
				_dumpAttributes(*this, out);
				out << "?>\n";
				break;
			}
			case hkXmlStreamParser::TOKEN_BLOCK_START_END:
			case hkXmlStreamParser::TOKEN_BLOCK_START:
			{
				out << "<";
				_dumpAttributes(*this, out);
				if (tok == hkXmlStreamParser::TOKEN_BLOCK_START_END)
				{
					out << "/";
				}
				out << ">\n";
				break;
			}
			case hkXmlStreamParser::TOKEN_BLOCK_END:
			{
				out << "</" << getBlockName() << ">\n";
				break;
			}
			case hkXmlStreamParser::TOKEN_TEXT:
			case hkXmlStreamParser::TOKEN_WHITESPACE:
			{
				out << getLexeme();
				break;
			}
			case hkXmlStreamParser::TOKEN_ERROR:
			case hkXmlStreamParser::TOKEN_EOF:
			{
				return;
			}
		}
	}
}

/* static */hkResult HK_CALL hkXmlStreamParser::parseInt(const hkSubString& subString, hkInt64& valueOut)
{
	const int maxLen = 31;

	if (subString.length() <= 0)
	{
		return HK_FAILURE;
	}

	// Make sure they are all digits
	const char* start = subString.m_start;
	const char* end = subString.m_end;

	const char* cur = start;
	HK_ASSERT(0x2423a432, end - start > 0);

	// Skip '-' if there is one
	if (*cur == '-') cur++;
	if (cur >= end) return HK_FAILURE;
	
	// Must be a start digit
	if (!hkXmlLexAnalyzer::isDigit(*cur))
	{
		return HK_FAILURE;
	}
	cur++;
	// Followed by digits
	for (; cur < end; cur++)
	{
		if (!hkXmlLexAnalyzer::isDigit(*cur))
		{
			return HK_FAILURE;
		}
	}

	// Looks good. Do the conversion
	char buffer[maxLen + 1];
	hkString::strNcpy(buffer, start, int(end - start));
	// Null terminate
	buffer[end - start] = 0;

	valueOut = hkString::atoll(buffer);
	return HK_SUCCESS;
}

hkResult HK_CALL hkXmlStreamParser::parseInt(const hkSubString& subString, int& signOut, hkUint64& magOut)
{
	const int maxLen = 31;
	signOut = 1;

	if (subString.length() <= 0)
	{
		return HK_FAILURE;
	}

	// Make sure they are all digits
	const char* start = subString.m_start;
	const char* end = subString.m_end;

	const char* cur = start;
	HK_ASSERT(0x2423a432, end - start > 0);

	// Account for '-' if there is one
	if (*cur == '-')
	{
		cur++;
		start++;
		signOut = -1;
	}

	if (cur >= end) return HK_FAILURE;

	// Hex:
	if ((end - cur > 2) && (cur[0] == '0') && ((cur[1] == 'x') || (cur[1] == 'X')))
	{
		for (cur = cur + 2; cur < end; ++cur)
		{
			if (!hkXmlLexAnalyzer::isHexDigit(*cur))
			{
				return HK_FAILURE;
			}
		}
	}
	else
	{
		// digits
		for (; cur < end; cur++)
		{
			if (!hkXmlLexAnalyzer::isDigit(*cur))
			{
				return HK_FAILURE;
			}
		}
	}
	
	// Looks good. Do the conversion
	char buffer[maxLen + 1];
	hkString::strNcpy(buffer, start, int(end - start));
	// Null terminate
	buffer[end - start] = 0;

	magOut = hkString::atoull(buffer);
	return HK_SUCCESS;
}

hkResult hkXmlStreamParser::getIntAttribute(const char* key, int& value) const
{
	hkSubString subString;
	if (getValue(key, subString) != HK_SUCCESS)
	{
		return HK_FAILURE;
	}

	if (!(subString.m_start[0] == '"' && subString.m_end[-1] == '"'))
	{
		return HK_FAILURE;
	}
	subString.m_start++;
	subString.m_end--;

	hkInt64 localValue = 0;
	hkResult res = parseInt(subString, localValue);
	value = static_cast<int>(localValue);
	return res;
}

hkBool hkXmlStreamParser::hasAttribute(const char* key) const
{
	hkSubString subString;
	return getValue(key, subString) == HK_SUCCESS;
}

/* static */hkResult HK_CALL hkXmlStreamParser::parseReal(const hkSubString& subString, hkReal& valueOut)
{
	const int maxLen = 32;

	if (subString.length() < 1 || subString.length() > maxLen )
	{
		return HK_FAILURE;
	}

	if (subString.m_start[0] == 'x')
	{
		// Its a hex
		hkUint64 v = 0;

		if ((subString.length() != 9) && (subString.length() != 17))
		{
			return HK_FAILURE;
		}

		const char* chars = subString.m_start + 1;
		for (int i = 0; i < subString.length()-1; i++)
		{
			v = v << 4;

			const char c = chars[i];
			if (c >= '0' && c <= '9')
			{
				v |= (c - '0');
			}
			else if (c >= 'A' && c <= 'F')
			{
				v |= (c - 'A') + 10;
			}
			else if (c >= 'a' && c <= 'f')
			{
				v |= (c - 'a') + 10;
			} 
			else
			{
				return HK_FAILURE;
			}
		}

		if (subString.length() == 9)
		{
			// its a float
			hkFloat32 v_lo = *(hkFloat32*)(((hkUint32*)&v)+HK_ENDIAN_BIG);
#if defined(HK_REAL_IS_DOUBLE)
			
			
			if (v_lo == 3.40282e+38f) valueOut = HK_REAL_MAX;
			else if (v_lo == FLT_MIN) valueOut = HK_REAL_MIN;
			else if (v_lo == FLT_EPSILON) valueOut = HK_REAL_EPSILON;
			else if (v_lo == 1.8446726e+019f) valueOut = HK_REAL_HIGH;
			else valueOut = hkReal(v_lo);
#else
			valueOut = hkReal(v_lo);
#endif
		}
		else if (subString.length() == 17)
		{
			// its a double
			hkDouble64 dval = *(hkDouble64*)&v;
#if !defined(HK_REAL_IS_DOUBLE)
			
			
			if (dval == 1.7970e+308) valueOut = HK_REAL_MAX;
			else if (dval == DBL_MIN) valueOut = HK_REAL_MIN;
			else if (dval == DBL_EPSILON) valueOut = HK_REAL_EPSILON;
			else if (dval == 1.8446726e+150) valueOut = HK_REAL_HIGH;
			else valueOut = hkReal(dval);
#else
			valueOut = hkReal(dval);
#endif
		}

		return HK_SUCCESS;
	}

	return hkFloatParseUtil::parseFloat(subString, valueOut);
}

hkResult hkXmlStreamParser::getRealAttribute(const char* key, hkReal& value) const
{
	hkSubString subString;
	if (getValue(key, subString) != HK_SUCCESS)
	{
		return HK_FAILURE;
	}

	if (!(subString.m_start[0] == '"' && subString.m_end[-1] == '"'))
	{
		return HK_FAILURE;
	}

	subString.m_start++;
	subString.m_end--;
	return parseReal(subString, value);
}

/* static */hkBool hkXmlStreamParser::needsDecode(const hkSubString& subString)
{
	const char* cur = subString.m_start;
	const char* end = subString.m_end;

	for (; cur < end; cur++)
	{
		if (*cur == '&')
		{
			return true;
		}
	}
	return false;
}

/* static */hkResult hkXmlStreamParser::decodeString(const hkSubString& subString, hkStringBuf& buf)
{
	buf.clear();

	const char* start = subString.m_start;
	const char* end = subString.m_end;

	while (start < end)
	{
		const char* cur = start;

		if (*cur == '&')
		{
			// Decode
			const int remaining = int(end - cur);

			if (remaining >= 5 && hkString::strNcmp(cur + 1, "amp;", 4) == 0)
			{
				buf.append("&", 1);
				start = cur + 5;
				continue;
			}
			if (remaining >= 4 && hkString::strNcmp(cur + 1, "lt;", 3) == 0)
			{
				buf.append("<", 1);
				start = cur + 4;
				continue;
			}
			if (remaining >= 4 && hkString::strNcmp(cur + 1, "gt;", 3) == 0)
			{
				buf.append(">", 1);
				start = cur + 4;
				continue;
			}
			if (remaining >= 6 && hkString::strNcmp(cur + 1, "quot;", 5) == 0)
			{
				buf.append("\"", 1);
				start = cur + 6;
				continue;
			}
			if (remaining >= 6 && hkString::strNcmp(cur + 1, "apos;", 5) == 0)
			{
				buf.append("'", 1);
				start = cur + 6;
				continue;
			}

			if (cur[1] == '#' && hkXmlLexAnalyzer::isDigit(cur[2]) && remaining >= 4)
			{
				char buffer[16];
				cur += 2;

				while (cur < end && hkXmlLexAnalyzer::isDigit(*cur)) 
				{
					cur++;
				}
				if (cur >= end || *cur != ';' || int(cur - start) > int(HK_COUNT_OF(buffer)))
				{
					// Didn't get closing ; or its not a digit
					return HK_FAILURE;
				}

				// Copy into the buffer
				hkString::strNcpy(buffer, start + 2, int(cur - (start + 2)));
				buffer[cur - (start + 2)] =0;

				int value = hkString::atoi(buffer);
			
				if (value < 0 || value > 0xff)
				{
					HK_ASSERT(0x3243a432, !"Value is too large to store");
					return HK_FAILURE;
				}
				
				buffer[0] = char(value);
				buffer[1] = 0;
				buf.append(buffer);

				// Step over the ';'
				cur++;
				start = cur;
				continue;
			}
			
			return HK_FAILURE;
		}
		else
		{
			cur = start + 1;
			
			while (cur < end && *cur != '&') 
			{
				cur++;
			}

			// Append it
			buf.append(start, int(cur - start));

			// Next
			start = cur;
		}
	}

	return HK_SUCCESS;
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
