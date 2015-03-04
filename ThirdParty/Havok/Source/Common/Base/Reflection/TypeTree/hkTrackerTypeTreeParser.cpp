/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Reflection/TypeTree/hkTrackerTypeTreeParser.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/Container/String/hkStringBuf.h>

/* static */hkBool HK_CALL hkTrackerTypeTreeParser::_parseTokens(const hkSubString& str, TokenContainer& container)
{
	const char* cur = str.m_start;
	const char* end = str.m_end;

	while (cur < end)
	{
		const char c = *cur++;
		switch (c)
		{
			case '[':
			{
				container.addToken(TOKEN_OPEN_SQUARE);
				break;
			}
			case ']':
			{
				container.addToken(TOKEN_CLOSE_SQUARE);
				break;
			}
			case ',':
			{
				container.addToken(TOKEN_COMMA);
				break;
			}
			case '<':
			{
				container.addToken(TOKEN_LT);
				break;
			}
			case '>':
			{
				container.addToken(TOKEN_GT);
				break;
			}
			case '*':
			{
				container.addToken(TOKEN_POINTER);
				break;
			}
			case '&':
			{
				container.addToken(TOKEN_REFERENCE);
				break;				
			}
			case ' ':
			{
				break;
			}
			case '(':
			{
				// we ran into a function or method pointer.
				hkUint8 counter = 1; // must go back to 0, add 1 for every '(' and remove 1 for every ')'
				while(cur < end && counter != 0)
				{
					if(*cur == '(')
					{
						++counter;
					}
					else if(*cur == ')')
					{
						--counter;
					}
					++cur;
				}

				// Move forward (in the type container we'll have the return type)
				// This is not perfect but shouldn't cause any issue
				break;
			}
			default:
			{
				if (c == '-' || _isDigit(c))
				{
					const char* start = cur - 1;
					while (cur < end && _isDigit(*cur)) cur++;
					
					container.addToken(TOKEN_INT_VALUE, hkSubString(start, cur));
					break;
				}

				if (c == '`') // `anonymous namespace'
				{
					// at least in MSVC, this opens with a backtick but closes with a single quote.
					// skip to the ending single-quote
					while (cur < end && (*cur) != '\'') { cur++; }
					if (cur[0] == '\'')
						cur++;
					break;
				}

				if (_isLetter(c) || c == '_' || c == ':')
				{
					const char* start = cur - 1;
					while (cur < end && 
						(_isLetter(*cur) || _isDigit(*cur) || *cur == ':' || *cur == '_')) 
					{
						cur++;
					}

					// If starts with :: - strip it out
					if ((cur - start) > 2 && start[0] == ':' && start[1] == ':')
					{
						start += 2;
					}

					hkSubString ident(start, cur);

					if (ident == "const")
					{
						// Ignore const
						break;
					}
					if (ident == "restrict")
					{
						// Ignore restrict
						break;
					}
					if (ident == "enum")
					{
						container.addToken(TOKEN_ENUM, ident);
						break;
					}
					if (ident == "struct" || ident == "class")
					{
						container.addToken(TOKEN_CLASS, ident);
						break;
					}
					if (ident == "unsigned")
					{
						container.addToken(TOKEN_UNSIGNED);
						break;
					}
					if (ident == "signed")
					{
						container.addToken(TOKEN_SIGNED);
						break;
					}

					Type builtInType = _calcBuiltInType(ident);
					if (builtInType != hkTrackerTypeTreeNode::TYPE_UNKNOWN)
					{
						container.addToken(TOKEN_BUILT_IN, ident, builtInType);
						break;
					}
					// Identifier
					container.addToken(TOKEN_IDENTIFIER, ident);
					break;
				}

				// Don't know what this is!
				return false;
			}
		}
	}

	container.addToken(TOKEN_TERMINATOR);
	return true;
}

/* static */const hkTrackerTypeTreeParser::Node* hkTrackerTypeTreeParser::_parseTerminalType(
	TokenRange& range, hkTrackerTypeTreeCache& cache, const hkStringBuf& prefix)
{
	if (range.m_start == range.m_end)
	{
		// Nothing there....
		return HK_NULL;
	}

	const Token* cur = range.m_start;
	switch (cur->m_token)
	{
		default:
		{
			return HK_NULL;
		}
		case TOKEN_SIGNED:
		case TOKEN_UNSIGNED:
		{
			range.m_start++;
			if (range.m_start >= range.m_end || range.m_start->m_token != TOKEN_BUILT_IN)
			{
				return HK_NULL;
			}

			Type type = range.m_start->m_builtInType;
			range.m_start++;

			if (cur->m_token == TOKEN_UNSIGNED)
			{
				// Make unsigned
				if (type >= hkTrackerTypeTreeNode::TYPE_INT8 && type <= hkTrackerTypeTreeNode::TYPE_INT64)
				{
					type = Type((type - hkTrackerTypeTreeNode::TYPE_INT8) + hkTrackerTypeTreeNode::TYPE_UINT8);
				}
			}
			else
			{
				// Make signed
				if (type >= hkTrackerTypeTreeNode::TYPE_UINT8 && type <= hkTrackerTypeTreeNode::TYPE_UINT64)
				{
					type = Type((type - hkTrackerTypeTreeNode::TYPE_UINT8) + hkTrackerTypeTreeNode::TYPE_INT8);
				}
			}
			return cache.getBuiltInType(type);
		}
		case TOKEN_BUILT_IN:
		{
			range.m_start++;
			return cache.getBuiltInType(cur->m_builtInType);
		}
		case TOKEN_ENUM:
		{
			if (cur + 2 > range.m_end || cur[1].m_token != TOKEN_IDENTIFIER)
			{
				return HK_NULL;
			}
			range.m_start = cur + 2;
			hkStringBuf name(prefix);
			name.append(cur[1].m_string.m_start, cur[1].m_string.length());
			return cache.newNamedNode(Node::TYPE_ENUM, name.cString(), true);
		}
		case TOKEN_IDENTIFIER:
		case TOKEN_CLASS:
		{
			const Node::Type type = (cur->m_token == TOKEN_IDENTIFIER) ? Node::TYPE_NAMED : Node::TYPE_CLASS;

			if (cur->m_token == TOKEN_CLASS)
			{
				// Skip the class token
				cur++;
			}


			if (cur >= range.m_end || cur->m_token != TOKEN_IDENTIFIER)
			{
				// Expecting the name
				return HK_NULL;
			}

			const Node* node;
			if (cur + 1 >= range.m_end || cur[1].m_token != TOKEN_LT)
			{
				// it is a non templated class
				range.m_start = cur + 1;
				hkStringBuf name(prefix);
				name.append(cur->m_string.m_start, cur->m_string.length());
				node = cache.newNamedNode(type, name.cString(), true);
			}
			else
			{
				// Class template 
				Node* tempNode = cache.newNode(hkTrackerTypeTreeNode::TYPE_CLASS_TEMPLATE);
				hkStringBuf name(prefix);
				name.append(cur->m_string.m_start, cur->m_string.length());
				tempNode->m_name = cache.newText(name.cString());
				cur++;

				HK_ASSERT(0x3243a432, cur->m_token == TOKEN_LT);
				// Skip the lt
				cur++;

				// Lets try and parse it
				const Node** prev = &tempNode->m_contains;

				// HACK. There are some ugly templates which have repeated parameters which 
				// fool the node sharing code. For this block put in some placeholders which 
				// defeat the sharing code and set them back to null on exit.
				// We may end up with some more duplications, but correct behavior.
				struct UndoPlaceHolders
				{
					~UndoPlaceHolders()
					{
						for(int i = 0; i < nodesWithPlaceholders.getSize(); ++i )
						{
							// if not overwritten, undo placeholder.
							if( *nodesWithPlaceholders[i] == reinterpret_cast<const Node*>(16) )
							{
								*nodesWithPlaceholders[i] = HK_NULL;
							}
						}
					}
					void addPlaceHolder(const Node** n)
					{
						*n = reinterpret_cast<const Node*>(16);
						nodesWithPlaceholders.pushBack(n);
					}
					hkArray<const Node**>::Temp nodesWithPlaceholders;
				};
				UndoPlaceHolders undo;

				range.m_start = cur;
				while (true)
				{
					const Node* subType = HK_NULL;

					if  (range.m_start < range.m_end && 
					     range.m_start->m_token == TOKEN_INT_VALUE)
					{
						// Its an int
						Node* curType = cache.newNode(hkTrackerTypeTreeNode::TYPE_INT_VALUE);
						curType->m_dimension = range.m_start->m_string.getInt();
						// Next
						range.m_start++;

						subType = curType;
					}

					if (!subType)
					{
						subType = _parseType(range, cache);	
					}
					if (!subType)
					{
						return HK_NULL;
					}

					// Okay lets see if its a built in cached one
					if (subType == cache.getBuiltInType(subType->m_type))
					{
						// Okay this is shared, so no good for our purposes. We need a unique type so 
						// we can use the m_next member
						subType = cache.newNode(subType->m_type);
					}

					// Attach to the list of types attached
					*prev = subType;
					prev = &subType->m_next;
					undo.addPlaceHolder(prev);

					if (range.m_start >= range.m_end)
					{
						// No comma or terminator
						return HK_NULL;
					}

					if (range.m_start->m_token == TOKEN_COMMA)
					{
						range.m_start ++;
						if (range.m_start >= range.m_end)
						{
							// No comma or terminator
							return HK_NULL;
						}
						continue;
					}

					if (range.m_start->m_token == TOKEN_GT)
					{
						// We are at the end
						range.m_start++;
						break;
					}
				}

				node = tempNode;
			}

			if( range.m_start < range.m_end &&
				(range.m_start->m_token == TOKEN_IDENTIFIER || range.m_start->m_token == TOKEN_CLASS) )
			{
				// The type is nested inside a template or a class (keep parsing)
				hkStringBuf newPrefix(prefix);
				{
					hkArray<char> ar;
					hkOstream os(ar);
					node->dumpType(os);
					newPrefix.append(ar.begin(), ar.getSize());
					newPrefix.append("::");
				}
				
				return _parseTerminalType(range, cache, newPrefix);
			}
			else
			{
				// Finished parsing the type (it is a template or a class)
				return node;
			}
		}
	}
}

/* static */const hkTrackerTypeTreeParser::Node* hkTrackerTypeTreeParser::_parseType(TokenRange& range, hkTrackerTypeTreeCache& cache)
{
	hkStringBuf emptyPrefix;
	const Node* node = _parseTerminalType(range, cache, emptyPrefix);
	if (!node)
	{
		return HK_NULL;
	}

	while (range.m_start < range.m_end)
	{
		const Token& token = *range.m_start;

		switch (token.m_token)
		{
			case TOKEN_POINTER:
			{
				Node* ptrNode = cache.newNode(hkTrackerTypeTreeNode::TYPE_POINTER);
				ptrNode->m_contains = node;
				node = ptrNode;

				range.m_start++;
				break;
			}
			case TOKEN_REFERENCE:
			{
				Node* ptrNode = cache.newNode(hkTrackerTypeTreeNode::TYPE_REFERENCE);
				ptrNode->m_contains = node;
				node = ptrNode;

				range.m_start++;
				break;
			}
			case TOKEN_OPEN_SQUARE:
			{
				if (range.m_start + 3 > range.m_end || 
					range.m_start[1].m_token != TOKEN_INT_VALUE || 
					range.m_start[2].m_token != TOKEN_CLOSE_SQUARE)
				{
					// Badly formed array
					return HK_NULL;
				}

				Node* arrayNode = cache.newNode(hkTrackerTypeTreeNode::TYPE_ARRAY);
				arrayNode->m_contains = node;

				// Set the size
				arrayNode->m_dimension = range.m_start[1].m_string.getInt();

				// 
				node = arrayNode;

				range.m_start += 3;
				break;
			}
			default:
			{
				return node;
			}
		}
	}

	return node;
}


/* static */hkTrackerTypeTreeParser::Type hkTrackerTypeTreeParser::_calcBuiltInType(const hkSubString& subString)
{
	if (subString.length() <= 0)
	{
		return hkTrackerTypeTreeNode::TYPE_UNKNOWN;
	}

	const char first = subString.m_start[0];

	switch (first)
	{
		case 'f':
		{
			if (subString == "float")
			{
				return hkTrackerTypeTreeNode::TYPE_FLOAT32;
			}
			break;
		}
		case 'd':
		{
			if (subString == "double")
			{
				return hkTrackerTypeTreeNode::TYPE_FLOAT64;
			}
			break;
		}
		case 's':
		{
			if (subString == "short")
			{
				return hkTrackerTypeTreeNode::TYPE_INT16;
			}
			break;
		}
		case 'i':
		{
			if (subString == "int")
			{
				return (sizeof(int) == 4) ? hkTrackerTypeTreeNode::TYPE_INT32 : hkTrackerTypeTreeNode::TYPE_INT64;
			}
			break;
		}
		case 'l':
		{
			if (subString == "long")
			{
				return (sizeof(long) == 4) ? hkTrackerTypeTreeNode::TYPE_INT32 : hkTrackerTypeTreeNode::TYPE_INT64;
			}
			break;
		}
		case 'h':
		{
			if (subString.length() < 2 || subString.m_start[1] != 'k')
			{
				break;
			}

			if (subString == "hkUint8")
			{
				return hkTrackerTypeTreeNode::TYPE_UINT8;
			}
			if (subString == "hkInt8")
			{
				return hkTrackerTypeTreeNode::TYPE_INT8;
			}
			if (subString == "hkUint16")
			{
				return hkTrackerTypeTreeNode::TYPE_UINT16;
			}
			if (subString == "hkInt16")
			{
				return hkTrackerTypeTreeNode::TYPE_INT16;
			}

			if (subString == "hkUint32")
			{
				return hkTrackerTypeTreeNode::TYPE_UINT32;
			}
			if (subString == "hkInt32")
			{
				return hkTrackerTypeTreeNode::TYPE_INT32;
			}

			if (subString == "hkUint64")
			{
				return hkTrackerTypeTreeNode::TYPE_UINT64;
			}
			if (subString == "hkInt64")
			{
				return hkTrackerTypeTreeNode::TYPE_INT64;
			}

			if (subString == "hkUlong")
			{
				return (sizeof(hkUlong) == 4) ? hkTrackerTypeTreeNode::TYPE_UINT32 : hkTrackerTypeTreeNode::TYPE_UINT64;
			}

			if (subString == "hkLong")
			{
				return (sizeof(hkLong)) == 4 ? hkTrackerTypeTreeNode::TYPE_INT32 : hkTrackerTypeTreeNode::TYPE_INT64;
			}

			if (subString == "hkReal")
			{
				return (sizeof(hkReal) == 4) ? hkTrackerTypeTreeNode::TYPE_FLOAT32 : hkTrackerTypeTreeNode::TYPE_FLOAT64;
			}

			if (subString == "hkFloat32")
			{
				return hkTrackerTypeTreeNode::TYPE_FLOAT32;
			}
			if (subString == "hkDouble64")
			{
				return hkTrackerTypeTreeNode::TYPE_FLOAT64;
			}

			break;
		}
		case '_':
		{
			if (subString == "__int8") 
			{
				return hkTrackerTypeTreeNode::TYPE_INT8;
			}
			if (subString == "__int16") 
			{
				return hkTrackerTypeTreeNode::TYPE_INT16;
			}
			if (subString == "__int32") 
			{
				return hkTrackerTypeTreeNode::TYPE_INT32;
			}
			if (subString == "__int64") 
			{
				return hkTrackerTypeTreeNode::TYPE_INT64;
			}

			if (subString == "__uint8") 
			{
				return hkTrackerTypeTreeNode::TYPE_UINT8;
			}
			if (subString == "__uint16") 
			{
				return hkTrackerTypeTreeNode::TYPE_UINT16;
			}
			if (subString == "__uint32") 
			{
				return hkTrackerTypeTreeNode::TYPE_UINT32;
			}
			if (subString == "__uint64") 
			{
				return hkTrackerTypeTreeNode::TYPE_UINT64;
			}
			break;
		}
		case 'c':
		{
			if (subString == "char")
			{
				return hkTrackerTypeTreeNode::TYPE_INT8;
			}

			break;
		}
		case 'v':
		{
			if (subString == "void")
			{
				return hkTrackerTypeTreeNode::TYPE_VOID;
			}
			break;
		}
		case 'b':
		{
			if (subString == "bool")
			{
				return hkTrackerTypeTreeNode::TYPE_BOOL;
			}
			break;
		}
		default: break;
	}

	return hkTrackerTypeTreeNode::TYPE_UNKNOWN;
}

/* static */const hkTrackerTypeTreeParser::Node* hkTrackerTypeTreeParser::parseNewType(const hkSubString& subString, hkTrackerTypeTreeCache& cache)
{
	TokenContainer container;
	hkBool ret = _parseTokens(subString, container);

	if(!ret)
	{
		// parsing failed (e.g. function pointer types)
		return HK_NULL;
	}

	TokenRange range;
	range.m_start = container.m_tokens;
	range.m_end = container.m_tokens + container.m_numTokens - 1;
	const Node* node = _parseType(range, cache);

	if (range.m_start == range.m_end)
	{
		// Must be at the end -> it's all parsed
		return node;
	}
	return HK_NULL;
}
/* static */const hkTrackerTypeTreeParser::Node* HK_CALL hkTrackerTypeTreeParser::parseType(const char* typeName, hkTrackerTypeTreeCache& cache)
{
	const Node* node;
	if (cache.getTypeExpressionTree(typeName, &node))
	{
		return node;
	}

	// Try parsing it 
	node = parseNewType(hkSubString(typeName), cache);

	// Add it to the cache
	cache.setTypeExpressionTree(typeName, node);
	return node;
}

/* static */const hkTrackerTypeTreeParser::Node* HK_CALL hkTrackerTypeTreeParser::parseType(const hkSubString& typeName, hkTrackerTypeTreeCache& cache)
{
	hkLocalBuffer<char> buffer(typeName.length() + 1);

	hkString::strNcpy(buffer.begin(), typeName.m_start, typeName.length());
	buffer[typeName.length()] = 0;

	return parseType(buffer.begin(), cache);
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
