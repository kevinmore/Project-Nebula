/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Util/Xml/hkXmlLexAnalyzer.h>

/* static */const char* const hkXmlLexAnalyzer::tokenNames[] = 
{
	"START_BLOCK", 
	"START_QBLOCK",
	"TEXT",		
	"WHITESPACE",
	"COMMENT",
	"EOF",		
	"END_BLOCK", 
	"END_QBLOCK",
	"IDENTIFIER",
	"EQUALS",
	"QUOTED_STRING",
	"SLASH",
	"ERROR",
};

HK_COMPILE_TIME_ASSERT(HK_COUNT_OF(hkXmlLexAnalyzer::tokenNames) == hkXmlLexAnalyzer::TOKEN_COUNT_OF);

hkXmlLexAnalyzer::hkXmlLexAnalyzer(hkStreamReader* reader):
	m_buffer(reader),
	m_state(0)
{
}

hkXmlLexAnalyzer::Token hkXmlLexAnalyzer::_lexComment()
{
    char match[] = "-->";
    int pos = 0;
    while (true)
	{
        char c = m_buffer.nextChar();
        if (c == 0)
		{
			return _handleError("Badly formed comment");
		}

        if (match[pos] == c)
        {
            pos++;
            if (pos == 3) 
			{
				return TOKEN_COMMENT;
			}
        }
        else
        {
            pos = 0;
        }
    }
}

hkXmlLexAnalyzer::Token hkXmlLexAnalyzer::_lexQuotedString()
{
	hkBool escaped = false;

	while (true)
	{
		const char c = m_buffer.nextChar();
		if (c == 0)
		{
			return _handleError("Didn't hit terminating \"");
		}

		if (escaped)
		{
			HK_ASSERT(0x242342, c != 0xa && c != 0xd);
			escaped = false;
			continue;
		}

		if (c == '\\')
		{
			escaped = true;
			continue;
		}

		if (c == '"')
		{
			return TOKEN_QUOTED_STRING;
		}
	}
}

hkXmlLexAnalyzer::Token hkXmlLexAnalyzer::_lexIdentifier()
{
	while (true)
	{
		const char c = m_buffer.getChar();
		if (isLetter(c) || isDigit(c) || c == '_' || c == ':')
		{
			m_buffer.nextChar();
		}
		else
		{
			return TOKEN_IDENTIFIER;
		}
	}
}

hkXmlLexAnalyzer::Token hkXmlLexAnalyzer::_lexWhiteSpace()
{
	while (true)
	{
		const char c = m_buffer.getChar();
		if (isWhiteSpace(c))
		{
			m_buffer.nextChar();
		}
		else
		{
			return TOKEN_WHITESPACE;
		}
	}
}

hkXmlLexAnalyzer::Token hkXmlLexAnalyzer::_lexText()
{
    m_buffer.rewindChar();
    while (true)
    {
        char c = m_buffer.nextChar();
        switch (c)
        {
            case 0:
			{
				return TOKEN_TEXT;
			}
            case '<':
            {
                m_buffer.rewindChar();
				return TOKEN_TEXT;
            }
            default:
            {
				if (isWhiteSpace(c))
				{
					m_buffer.rewindChar();
					return TOKEN_TEXT;
				}
				break;
            }
        }
    }
}


hkXmlLexAnalyzer::Token hkXmlLexAnalyzer::_matchInBrackets()
{
	while (true)
	{
		char c = m_buffer.nextChar();
		switch (c)
		{
			case 0:
			{
				return TOKEN_EOF;
			}
			case '=':
			{
				return TOKEN_EQUALS;
			}
			case ' ':
			case '\t':
			case '\n':
			case '\r':
			{
				_lexWhiteSpace();
				m_buffer.lexemeCommit();
				// Just ignore this white space
				break;
			}
			case '?':
			{
				if (m_state & IN_QBLOCK)
				{
					// Read the closing > 
					char nextC = m_buffer.nextChar();
					if (nextC == '>')
					{
						m_state &= ~(IN_BRACKETS | IN_QBLOCK);
						return TOKEN_END_QBLOCK;
					}
					m_buffer.rewindChar();
				}
				break;
			}
			case '>':
			{
				if (m_state & IN_QBLOCK)
				{
					return _handleError("Expecting ?> to close <? section");
				}
				
				m_state &= ~IN_BRACKETS;
				return TOKEN_END_BLOCK;
			}
			case '/':
			{
				return TOKEN_SLASH;
			}
			case '\"':
			{
				// Quoted string 
				return _lexQuotedString();
			}
			default:
			{
				if (isLetter(c) || c == '_' || c == ':')
				{
					return _lexIdentifier();
				}
				return _handleError("Unexpected token");
			}
		}
	}
}


hkXmlLexAnalyzer::Token hkXmlLexAnalyzer::_matchOutsideBrackets()
{
    char c = m_buffer.nextChar();
    switch (c)
    {
        case 0:
		{
			return TOKEN_EOF;
		}
        case '<':
        {
			if (m_buffer.getChar() == '?')
			{
				m_buffer.nextChar();
				m_state |= (IN_BRACKETS | IN_QBLOCK);
				return TOKEN_START_QBLOCK;
			}
			if (m_buffer.matchAndConsume("!--"))
			{
				return _lexComment();
			}
			else
			{
				m_state |= IN_BRACKETS;
				return TOKEN_START_BLOCK;
			}
        }
        default:
        {
			if (isWhiteSpace(c))
			{
				return _lexWhiteSpace();
			}
			else
			{
				return _lexText();
			}
        }
    }
}

hkXmlLexAnalyzer::Token hkXmlLexAnalyzer::_handleError(const char* desc)
{
	HK_WARN(0x2343243, desc);
	return TOKEN_ERROR;
}

hkXmlLexAnalyzer::Token hkXmlLexAnalyzer::advance()
{
	m_buffer.lexemeCommit();
    if (m_state & IN_BRACKETS)
    {
        return _matchInBrackets();
    }
    else
    {
        return _matchOutsideBrackets();
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
