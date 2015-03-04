/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Tagfile/Text/hkTextTagfileWriter.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/System/Io/Writer/hkStreamWriter.h>
#include <Common/Serialize/Data/hkDataObject.h>

template <typename Value>
static hkBool32 worthPrinting(const Value& value)
{
	hkDataObject::Type type = value.getType();

	if( type->isArray())
	{
		return value.asArray().getSize();
	}
	else if( type->isTuple())
	{
		if (type->getParent()->isReal())
		{
			// Vecs always printed 
			return true;
		}

		hkDataArray a = value.asArray();
		int asize = a.getSize();
		for( int i = 0; i < asize; ++i )
		{
			if( worthPrinting(a[i]) )
			{
				return true;
			}
		}
		return false;
	}
	
	switch( type->getSubType())
	{
		case hkTypeManager::SUB_TYPE_BYTE:
		case hkTypeManager::SUB_TYPE_INT:
		{
			return value.asInt() != 0;
		}
		case hkTypeManager::SUB_TYPE_REAL:
		{
			return value.asReal() != 0;
		}
		case hkTypeManager::SUB_TYPE_POINTER:
		case hkTypeManager::SUB_TYPE_CLASS:
		{
			return value.asObject().isNull() == hkFalse32;
		}
		case hkTypeManager::SUB_TYPE_CSTRING:
		{
			const char* s = value.asString();
			return s != 0;
		}
		case hkTypeManager::SUB_TYPE_VOID: // for +nosave
		{
			return false;
		}
		default:
		{
			HK_ASSERT(0x43e4b8f8, 0);
			return false;
		}
	}
	//return true;
}

static int calcWrapMod(hkDataObject::Type type)
{
	switch (type->getSubType())
	{
		case hkTypeManager::SUB_TYPE_VOID:	return 32;
		case hkTypeManager::SUB_TYPE_BYTE:	return 16;
		case hkTypeManager::SUB_TYPE_INT:	return  8;
		case hkTypeManager::SUB_TYPE_REAL:	return  8; 
		case hkTypeManager::SUB_TYPE_CSTRING:	return 1;
		case hkTypeManager::SUB_TYPE_POINTER: return 1;
		case hkTypeManager::SUB_TYPE_CLASS:	return -1;
		default:							return 1;
	}
}

static void writeFloat( hkOstream& os, hkReal r )
{
	hkStringBuf s; s.printf("%g", r);
	if( s.indexOf('.') == -1 ) // make sure we have a decimal point
	{
		s += ".0";
	}
	os.write( s.cString(), s.getLength() );
}

static void writeReals( hkOstream& os, const hkReal* ra, int nreal )
{
	os.printf("<");
	for( int i = 0; i < nreal; ++i )
	{
		if( i != 0 ) os.printf(", ");
		writeFloat( os, ra[i] );
	}
	os.printf(">");
}


struct Context
{
	struct HandleOps
	{
		inline static unsigned hash( const hkDataObject::Handle& key, unsigned mod )
		{
			return unsigned((hkUlong(key.p0) >> 4) * 2654435761U) & mod;
		}
		inline static void invalidate( hkDataObject::Handle& key )
		{
			key.p0 = (void*)hkUlong(-1);
		}
		inline static hkBool32 isValid( hkDataObject::Handle& key )
		{
			return key.p0 != (void*)hkUlong(-1);
		}
		inline static hkBool32 equal( const hkDataObject::Handle& key0, const hkDataObject::Handle& key1 )
		{
			return key0.p0 == key1.p0 && key0.p1 == key1.p1;
		}
	};
	typedef hkMap<hkDataObject::Handle, int, HandleOps> DoneObjects;

	struct Details
	{
		void clear() { refCount = 0; written = false; }
		int refCount;
		hkBool32 written;
	};

	Context(hkStreamWriter* sw, hkTagfileWriter::AddDataObjectListener* listener) : m_indentStep(1), m_realWriter(sw), m_os(sw), m_listenerCache(listener)
	{
		m_indent.pushBack(0); m_indent.popBack();
 		hkDataObject::Handle zero = {0,0};
 		m_done.insert(zero, 0);
 		m_details.expandOne().clear();
 		m_details[0].written = true;
	}
	void pushIndent()
	{
		m_indent.setSize( m_indent.getSize() + m_indentStep, ' ' );
		m_indent.pushBack(0); m_indent.popBack();
	}
	void popIndent()
	{
		m_indent[ m_indent.getSize() - m_indentStep] = 0;
		m_indent.setSize( m_indent.getSize() - m_indentStep );
	}
	const char* getIndent() const
	{
		return m_indent.begin();
	}
	
	int getObjectIndex(const hkDataObject& obj) const
	{
		return m_done.getWithDefault(obj.getHandle(), -1 );
	}

	bool isFirstTimeSeeing(const hkDataObject& obj)
	{
		bool firstTime = false;
		hkDataObject::Handle oh = obj.getHandle();
		int idx = m_done.getWithDefault(oh,-1);
		if( idx == -1 )
		{
			idx = m_details.getSize();
			m_done.insert(oh, idx);
			m_details.expandOne().clear();
			firstTime = true;
		}
		m_details[idx].refCount += 1;
		return firstTime;
	}

	void scanObject(const hkDataObject& obj)
	{
		const hkDataObject objToScan = m_listenerCache.callListenerAndCache(obj);
		if( objToScan.isNull() == hkFalse32 && isFirstTimeSeeing(objToScan) )
		{
			scanObjectMembers(objToScan);
		}
	}

	void scanObjectMembers(const hkDataObject& obj)
	{
		for( hkDataObject::Iterator it = obj.getMemberIterator();
			obj.isValid(it);
			it = obj.getNextMember(it) )
		{
			hkDataObject::Value value = obj.getMemberValue(it);

			hkDataObject::Type type = value.getType();
			hkDataObject::Type term = type->findTerminal();
			if( term->isClass())
			{
				switch( type->getSubType() )
				{
					case hkTypeManager::SUB_TYPE_CLASS:
					{
						scanObjectMembers( value.asObject() );
						break;
					}
					case hkTypeManager::SUB_TYPE_POINTER:
					{
						scanObject(value.asObject());
						break;
					}
					case hkTypeManager::SUB_TYPE_ARRAY:
					case hkTypeManager::SUB_TYPE_TUPLE:
					{
						hkDataObject::Type parent = type->getParent();
						if (parent->isClassPointer())
						{
							hkDataArray a = value.asArray();
							int arraySize = a.getSize();
							for( int i = 0; i < arraySize; ++i )
							{
								scanObject( a[i].asObject() );
							}
							break;
						}
						else if (parent->isClass())
						{
							hkDataArray a = value.asArray();
							int arraySize = a.getSize();
							// todo: check if the class has any pointers and we could skip this block
							for( int i = 0; i < arraySize; ++i )
							{
								// structs can't be pointed to and there are generally
								// lots of them, so we don't save refcount info for them
								scanObjectMembers( a[i].asObject() );
							}
							break;
						}
						HK_ASSERT(0x2343432, !"Unable to handle type");
					}
					default:
					{
						HK_ASSERT(0x6c5caf8e, 0);
					}
				}
			}
		}
	}

	template <typename Value>
	void writeValue(const Value& value)
	{
		hkDataObject::Type type = value.getType();

		switch( type->getSubType() )
		{
			case hkTypeManager::SUB_TYPE_VOID://XXX should not happen
			{
				HK_ASSERT(0x7fe599dc, 0);
				break;
			}
			case hkTypeManager::SUB_TYPE_BYTE:
			case hkTypeManager::SUB_TYPE_INT:
			{
				m_os.printf("%i", value.asInt());
				break;
			}
			case hkTypeManager::SUB_TYPE_REAL:
			{
				writeFloat(m_os, static_cast<hkReal>(value.asReal()));
				break;
			}
			case hkTypeManager::SUB_TYPE_POINTER:
			{
				writeObject( value.asObject() );
				break;
			}
			case hkTypeManager::SUB_TYPE_CLASS:
			{
				m_os.printf("{ ");
				writeObjectMembers( value.asObject() );
				m_os.printf(" }");
				break;
			}
			case hkTypeManager::SUB_TYPE_CSTRING:
			{
				m_os.printf("\"%s\"", value.asString());
				break;
			}
			case hkTypeManager::SUB_TYPE_TUPLE:
			{
				if (type->getParent()->isReal())
				{
					const int numReals = type->getTupleSize();
					writeReals( m_os, static_cast<const hkReal *>(value.asVec(numReals)), numReals);
					break;
				}
				// fall thru
			}
			case hkTypeManager::SUB_TYPE_ARRAY:
			{
				if (type->getParent()->isClass())
				{
					hkDataArray a = value.asArray();
					pushIndent();
					m_os.printf("[");
					hkDataClass k = a.getClass();
					hkArray<hkDataClass::MemberInfo>::Temp minfos(k.getNumMembers());
					k.getAllMemberInfo(minfos);
					for( int mi = 0; mi < minfos.getSize(); ++mi )
					{
						if( minfos[mi].m_type->isVoid())
						{
							continue;
						}
						hkDataArray a0 = a.swizzleObjectMember( minfos[mi].m_name );
						m_os.printf("\n%s%s : ", m_indent.begin(), minfos[mi].m_name);
						writeArray(a0);
					}
					m_os.printf("]");
					popIndent();
					break;
				}
				pushIndent();
				writeArray(value.asArray());
				popIndent();
				break;
			}			
			default:
				HK_ASSERT(0x43e4b8f8, 0);
		}
	}

	void writeArray( const hkDataArray& a )
	{
		m_os.printf("[");
		
		const int wrapCounterInit = calcWrapMod(a.getType());
		int wrapCounter = wrapCounterInit;
		int arraySize = a.getSize();
		for( int i = 0; i < arraySize; ++i )
		{
			writeValue( a[i] );
			if( i != arraySize-1 )
			{
				m_os.printf(", ");
				if( --wrapCounter == 0 )
				{
					m_os.printf("\n%s", getIndent());
					wrapCounter = wrapCounterInit;
				}
			}
		}
		m_os.printf("]");
	}

	void writeObjectMembers( const hkDataObject& obj )
	{
		pushIndent();
		const char* comma = ""; // no comma first time round
		for( hkDataObject::Iterator it = obj.getMemberIterator();
			obj.isValid(it);
			it = obj.getNextMember(it) )
		{
			hkDataObject::Value value = obj.getMemberValue(it);
			if( worthPrinting(value) )
			{
				m_os.printf("%s\n%s%s : ", comma, getIndent(), obj.getMemberName(it) );
				comma = ",";
				writeValue(value);
			}
		}
		popIndent();
	}

	void writeObject( const hkDataObject& obj )
	{
		const hkDataObject objToSave = m_listenerCache.getCachedDataObject(obj);
		if( objToSave.isNull() )
		{
			m_os.printf("*_0000");
			return;
		}
		int objectIndex = getObjectIndex(objToSave);
		if( objectIndex != -1 )
		{
			Context::Details& details = m_details[objectIndex];
			if( details.written )
			{
				m_os.printf("*_%04i", objectIndex );
				return;
			}
			m_os.printf("\n%s", getIndent());
			if( details.refCount > 1 )
			{
				m_os.printf("&_%04i ", objectIndex );
			}
			m_os.printf("{ " );
			details.written = true;

			hkDataClass klass = objToSave.getClass();
			m_os.printf("(%s, %i)", klass.getName(), klass.getVersion() );
		}
		else
		{
			m_os.printf("\n%s{ ", getIndent());
		}

		writeObjectMembers(objToSave);
		m_os.printf(" }");
	}

	class ListenerCache
	{
		public:

			ListenerCache(hkTagfileWriter::AddDataObjectListener* listener) : m_listener(listener)
			{
			}

			~ListenerCache()
			{
				for( int i = 0; i < m_cachedObjects.getSize(); ++i )
				{
					delete m_cachedObjects[i];
				}
			}

			hkDataObject getCachedDataObject(const hkDataObject& object)
			{
				if( m_listener && !object.isNull() )
				{
					hkDataObject::Handle h = object.getHandle();
					int index = m_indexFromHandle.getWithDefault(h, -1);
					if( index != -1 )
					{
						return m_cachedObjects[index]->getImplementation();
					}
				}
				return object;
			}

			hkDataObject callListenerAndCache(const hkDataObject& object)
			{
				if( m_listener && !object.isNull() )
				{
					hkDataObject::Handle h = object.getHandle();
					HK_ASSERT(0x36ed57e1, h.p0);
					int index = m_indexFromHandle.getWithDefault(h, -1);
					if( index == -1 )
					{
						hkDataObject newObject = m_listener->addDataObjectCallback(object);
						if( newObject != object )
						{
							m_indexFromHandle.insert(h, m_cachedObjects.getSize());
							m_cachedObjects.expandOne() = new hkDataObject(newObject);
							return newObject;
						}
					}
					else
					{
						return m_cachedObjects[index]->getImplementation();
					}
				}
				return object;
			}

		private:

			hkTagfileWriter::AddDataObjectListener* m_listener;
			DoneObjects m_indexFromHandle;
			hkArray<hkDataObject*> m_cachedObjects;
	};

	const int m_indentStep;
	hkArray<char> m_indent;
	hkArray<Details> m_details;
	DoneObjects m_done;

	hkStreamWriter* m_realWriter;
	hkArray<char> m_buf;
	hkOstream m_os;
	ListenerCache m_listenerCache;
};
 
hkResult hkTextTagfileWriter::save( const hkDataObject& obj, hkStreamWriter* stream, hkTagfileWriter::AddDataObjectListener* userListener, const Options& options )
{
	Context context(stream, userListener);
	context.scanObject(obj);
 	context.writeObject(obj);
	stream->flush();
	return HK_SUCCESS;
}

// explicitly instantiate our map type
#include <Common/Base/Container/PointerMap/hkMap.cxx>
template class hkMapBase<hkDataObject::Handle, int, Context::HandleOps>;
template class hkMap<hkDataObject::Handle, int, Context::HandleOps>;

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
