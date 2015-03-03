/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Tagfile/Xml/hkXmlTagfileCommon.h>
#include <Common/Serialize/Tagfile/Xml/hkXmlTagfileWriter.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/System/Io/Writer/hkStreamWriter.h>
#include <Common/Serialize/Data/hkDataObject.h>
#include <Common/Serialize/Util/Xml/hkFloatParseUtil.h>
#include <Common/Base/Config/hkConfigVersion.h>

namespace
{
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
			if (type->isVec())
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
				return value.asInt64() != 0;
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

	static const char* nameFromType(hkDataObject::Type type)
	{
		switch (type->getSubType())
		{
			case hkTypeManager::SUB_TYPE_VOID:	return "void";
			case hkTypeManager::SUB_TYPE_BYTE:	return "byte";
			case hkTypeManager::SUB_TYPE_INT:	return "int";
			case hkTypeManager::SUB_TYPE_REAL:	return "real";
			case hkTypeManager::SUB_TYPE_CSTRING:	return "string";
			case hkTypeManager::SUB_TYPE_POINTER: return "ref";
			case hkTypeManager::SUB_TYPE_CLASS:	return "struct";
			case hkTypeManager::SUB_TYPE_TUPLE:
			{
				if (type->getParent()->isReal())
				{
					switch (type->getTupleSize())
					{
						case 4: return "vec4";
						case 8: return "vec8";
						case 12: return "vec12";
						case 16: return "vec16";
						default: break;
					}
				}
				break;
			}
			default: break;
		}

		return HK_NULL;
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

	class Indenter
	{
		public:

			Indenter(int step = 2, char what=' ')
				: m_indentStep(step), m_value(what)
			{
				m_indent.pushBack(0);
				m_indent.popBack();
			}
			void push()
			{
				m_indent.setSize( m_indent.getSize() + m_indentStep, m_value );
				m_indent.pushBack(0);
				m_indent.popBack();
			}
			void pop()
			{
				m_indent[ m_indent.getSize() - m_indentStep] = 0;
				m_indent.setSize( m_indent.getSize() - m_indentStep );
			}
			const char* get() const
			{
				return m_indent.begin();
			}

		private:

			hkArray<char> m_indent;
			const int m_indentStep;
			char m_value;
	};

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

	class SeenObjects
	{
		public:

			SeenObjects(hkTagfileWriter::AddDataObjectListener* listener) : m_listener(listener)
			{
				// we always see a null object at index zero
				hkDataObject::Handle null = { HK_NULL, HK_NULL };
				m_objects.pushBack(null);
				m_objectIndexFromHandle.insert(null, 0);
			}

			void scanObject(const hkDataObject& objIn)
			{
				if( objIn.isNull() )
				{
					return;
				}

				if( m_objectIndexFromHandle.getWithDefault(objIn.getHandle(),-1) == -1 )
				{
					const hkDataObject obj = callListenerAndCache(objIn);

 					if( objIn.isNull() )
 					{
 						return;
 					}

					addClass( obj.getClass() ); // maybe we can skip metadata?

					int objIndex = m_objects.getSize();
					m_objects.pushBack( obj.getHandle() );
					m_objectIndexFromHandle.insert( objIn.getHandle(), objIndex);
					if( obj.getHandle() != objIn.getHandle() )
					{
						m_objectIndexFromHandle.insert( obj.getHandle(), objIndex);
					}
					scanObjectMembers(obj);
				}
			}

			int getObjectId(const hkDataObject& obj) const
			{
				return m_objectIndexFromHandle.getWithDefault(obj.getHandle(), -1);
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

			hkDataObject callListenerAndCache(const hkDataObject& object)
			{
				HK_ASSERT(0x386312fd, !object.isNull() );
				HK_ASSERT(0x1c2df479, m_objectIndexFromHandle.hasKey(object.getHandle()) == false );
				if( m_listener )
				{
					hkDataObject newObject = m_listener->addDataObjectCallback(object);
					if( newObject != object )
					{
						m_cachedObjects.expandOne() = newObject;
						return newObject;
					}
				}
				return object;
			}

			void addClass( const hkDataClass& k )
			{
				HK_ASSERT(0x4b183dc0, k.isNull()==hkFalse32);
				if( m_classIndexFromName.getWithDefault(k.getName(), -1) == -1 )
				{
					if( k.getParent().isNull() == hkFalse32 )
					{
						addClass(k.getParent());
					}

					m_classIndexFromName.insert(k.getName(), m_classes.getSize());
					m_classes.pushBack( k );
					
					hkArray<hkDataClass::MemberInfo>::Temp minfos; minfos.setSize(k.getNumDeclaredMembers());
					k.getAllDeclaredMemberInfo(minfos);
					for( int i = 0; i < minfos.getSize(); ++i )
					{
						const hkDataClass::MemberInfo& m = minfos[i];
						hkDataObject::Type term = m.m_type->findTerminal();
						if( term->isClass())
						{
							const hkDataClassImpl* clsImpl = k.getWorld()->findClass(term->getTypeName());
							HK_ASSERT(0x242343aa, clsImpl);

							hkDataClass c(const_cast<hkDataClassImpl*>(clsImpl));
							addClass(c);
						}
					}
				}
			}
			const hkArray<hkDataClass>& getClasses() const
			{
				return m_classes;
			}

			const hkArray<hkDataObject::Handle>& getObjects() const
			{
				return m_objects;
			}

		private:

			struct Details
			{
				Details()
				{
					m_handle.p0 = HK_NULL;
					m_handle.p1 = HK_NULL;
					m_index = 0;
				}
				
				hkDataObject::Handle m_handle;
				int m_index;
			
			};
			typedef hkMap<hkDataObject::Handle, int, HandleOps> ObjectMap;
			typedef hkStringMap<int> ClassMap;

			// We're careful here to only store object handles: that is two pointers from which 
			// we can reconstruct a dataobject. This means that we have a lower memory footprint
			// when using hkDataObjectNative - we don't have to keep the temp wrappers alive, just
			// the variant to reconstruct them as needed.
			ObjectMap m_objectIndexFromHandle;
			hkArray<hkDataObject::Handle> m_objects;

			// The objects returned from the listener can't have this optimization, we need to store
			// them here to keep them alive, since they're not owned by anything else.
			hkTagfileWriter::AddDataObjectListener* m_listener;
			hkArray<hkDataObject> m_cachedObjects;

			hkArray<hkDataClass> m_classes;
			ClassMap m_classIndexFromName;
	};

	struct Context
	{
		Context(hkTagfileWriter::AddDataObjectListener* listener, const hkTagfileWriter::Options& options) : 
			m_indenter(2), 
			m_seenObjects(listener)
		{
			m_writeFloatAsHex = options.m_exact;
			m_floatComment = options.m_verbose;
		}
		
		void writeClass(const hkDataClass& k, hkOstream& os)
		{
			if( k.getVersion() < 0 )
			{
				HK_WARN(0x6c482a3d, "Serializing class " << k.getName() << " which is marked as under development (negative version number). "
					"No patching will be done when loading which means it may load incorrectly in the future.");
			}
			os.printf("%s<class name=\"%s\" version=\"%i\"", m_indenter.get(), k.getName(), k.getVersion());
			if( k.getParent().getImplementation() )
			{
				os.printf(" parent=\"%s\"", k.getParent().getName());
			}
			os.printf(">\n");

			m_indenter.push();
			hkArray<hkDataClass::MemberInfo>::Temp minfos; minfos.setSize(k.getNumDeclaredMembers());
			k.getAllDeclaredMemberInfo(minfos);
			for( int i = 0; i < minfos.getSize(); ++i )
			{
				hkDataClass::MemberInfo& m = minfos[i];

				hkDataObject::Type type = m.m_type;
				hkBool isArray = false;
				int tupleCount = 0;

				if(type->isArray())
				{
					isArray = true;
					type = type->getParent();
				}
				else if(type->isTuple())
				{
					if(!type->isVec())
					{
						tupleCount = type->getTupleSize();
						type = type->getParent();
					}
				}
				
				os.printf("%s<member name=\"%s\" type=\"%s\"", m_indenter.get(), m.m_name, nameFromType(type));
				if( isArray)
				{
					os.printf(" array=\"true\"");
				}
				if( tupleCount)
				{
					os.printf(" count=\"%i\"", tupleCount);
				}

				hkDataObject::Type term = type->findTerminal();
				if( term->isClass())
				{
					os.printf(" class=\"%s\"", term->getTypeName());
				}
				os.printf("/>\n");
			}
			m_indenter.pop();
			os.printf("%s</class>\n", m_indenter.get());
		}

		void setContents(const hkDataObject& obj)
		{
			m_seenObjects.scanObject(obj);
		}

		void writeFloat( hkOstream& os, hkReal r )
		{
			hkStringBuf s;

			if (m_writeFloatAsHex)
			{
#if defined(HK_REAL_IS_DOUBLE)
				const hkUint64 v = *(hkUint64*)&r;
				s.printf("x%016" HK_PRINTF_FORMAT_INT64_SIZE "x", v);
#else
				const hkUint32 v = *(hkUint32*)&r;
				s.printf("x%08x", v);
#endif
				if (m_floatComment)
				{
					hkStringBuf b;
					hkFloatParseUtil::calcFloatTextWithPoint(r, b);	

					s.append(" <!-- ");
					s.append(b.cString());
					s.append(" -->");
				}
			}
			else
			{	
				hkFloatParseUtil::calcFloatTextWithPoint(r, s);	
			}

			os.write( s.cString(), s.getLength() );
		}
		void writeReals( hkOstream& os, const hkReal* ra, int nreal )
		{
			for( int i = 0; i < nreal; ++i )
			{
				if( i != 0 ) 
				{
					os.printf(" ");
				}
				writeFloat( os, ra[i] );
			}
		}

		template <typename Value>
		void writeValue(const Value& value, hkOstream& os)
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
					os.printf(HK_PRINTF_FORMAT_INT64, value.asInt64());
					break;
				}
				case hkTypeManager::SUB_TYPE_REAL:
				{							
					writeFloat(os, value.asReal());
					break;
				}
				case hkTypeManager::SUB_TYPE_POINTER:
				{
					os.printf("#%04i", m_seenObjects.getObjectId(value.asObject()) );
					break;
				}
				case hkTypeManager::SUB_TYPE_CLASS:
				{
					writeObjectMembers( value.asObject(), os );
					os.printf("\n%s", m_indenter.get());
					break;
				}
				case hkTypeManager::SUB_TYPE_CSTRING:
				{
					if( const char* sorig = value.asString() )
					{
						hkStringBuf sb(sorig);
						sb.replace("&", "&amp;", hkStringBuf::REPLACE_ALL);
						sb.replace("<", "&lt;", hkStringBuf::REPLACE_ALL);
						sb.replace(">", "&gt;", hkStringBuf::REPLACE_ALL);
						os.write(sb.cString(), sb.getLength());
					}
					else
					{
						os.printf("<null/>");
					}
					break;
				}
				case hkTypeManager::SUB_TYPE_TUPLE:
				{
					if (type->isVec())
					{
						const int numReals = type->getTupleSize();
						writeReals( os, static_cast<const hkReal *>(value.asVec(numReals)), numReals);
						break;
					}
					// fall thru
				}
				case hkTypeManager::SUB_TYPE_ARRAY:
				{
					hkDataObject::Type parent = type->getParent();
					if (parent->isClass())
					{
						hkDataArray arr = value.asArray();
						m_indenter.push();
						for( int si = 0; si < arr.getSize(); ++si )
						{
							os.printf("\n%s<struct>", m_indenter.get());
							writeObjectMembers( arr[si].asObject(), os );
							os.printf("\n%s</struct>", m_indenter.get());
						}
						m_indenter.pop();
						os.printf("\n%s", m_indenter.get());
						break;
					}

					m_indenter.push();
					writeArray(value.asArray(), os);
					m_indenter.pop();
					os.printf("\n%s", m_indenter.get());
					break;
				}
				default:
					HK_ASSERT(0x43e4b8f8, 0);
			}
		}

		void writeArray( const hkDataArray& a, hkOstream& os )
		{
			const hkDataObject::Type t = a.getType();

			int wrapCounterInit = calcWrapMod(t);
			int wrapCounter = 1;
			int arraySize = a.getSize();
			const char* itemType = nameFromType(t);

 			{	
 				if( t->isInt() || t->isReal() || t->isByte())
 				{
 					itemType = HK_NULL; // for readability we don't write <int>0</int><int>1</int> ... for basic types
 				}
 			}

			for( int i = 0; i < arraySize; ++i )
			{
				if( --wrapCounter == 0 )
				{
					os.printf("\n%s", m_indenter.get());
					wrapCounter = wrapCounterInit;
				}
				if( itemType )
				{
					if (t->isCstring() && a[i].asString() == HK_NULL)
					{
						os.printf("<null/>");
					}
					else
					{
						os.printf("<%s>", itemType );
						writeValue( a[i], os );
						os.printf("</%s>", itemType );
					}
				}
				else
				{
					writeValue( a[i], os );
					os.printf(" ");
				}
			}
		}

		void writeObjectMembers( const hkDataObject& obj, hkOstream& os )
		{
			m_indenter.push();
			for( hkDataObject::Iterator it = obj.getMemberIterator();
				obj.isValid(it);
				it = obj.getNextMember(it) )
			{
				hkDataObject::Value value = obj.getMemberValue(it);
				hkDataObject::Type type = value.getType();

				if( worthPrinting(value) )
				{
					if (type->isArray() || (type->isTuple() && !type->isVec()))
					{
						const char* memType = (type->isTuple()) ? "tuple" : "array";
						int size = (type->isTuple()) ? type->getTupleSize() : value.asArray().getSize();
						
						os.printf("\n%s<%s name=\"%s\" size=\"%i\">", m_indenter.get(), memType, obj.getMemberName(it), size );
						writeValue(value, os);
						os.printf("</%s>", memType);
					}
					else
					{
						const char* memType = nameFromType(type);

						if (type->isCstring() && value.asString() == HK_NULL)
						{
							os.printf("\n%s<null name=\"%s\"/>", m_indenter.get(), obj.getMemberName(it) );
						}
						else
						{
							os.printf("\n%s<%s name=\"%s\">", m_indenter.get(), memType, obj.getMemberName(it) );
							writeValue(value, os);
							os.printf("</%s>", memType);
						}
					}
				}
			}
			m_indenter.pop();
		}

		void writeObject( const hkDataObject& obj, hkOstream& os )
		{
			HK_ASSERT(0x1a44486b, obj.isNull()==hkFalse32);
			int objId = m_seenObjects.getObjectId(obj);
			const char* className = obj.getClass().getName();

			os.printf("\n%s<object id=\"#%04i\" type=\"%s\">", m_indenter.get(), objId, className );
			writeObjectMembers(obj, os);
			os.printf("\n%s</object>", m_indenter.get());
		}

		void writeFile( const hkDataObject& topObj, hkStreamWriter* stream)
		{
			hkOstream os(stream);
			os.printf("<?xml version=\"1.0\" encoding=\"ascii\"?>");
			os.printf("\n<hktagfile version=\"%d\" sdkversion=\"%s\">\n", hkXmlTagfile::XML_TAGFILE_VERSION, HAVOK_SDK_VERSION_STRING);
			m_indenter.push();

			const hkArray<hkDataClass>& classes = m_seenObjects.getClasses();
			for( int i = 0; i < classes.getSize(); ++i )
			{
				writeClass(classes[i], os);
			}

			const hkDataWorld* world = topObj.getClass().getWorld();
			const hkArray<hkDataObject::Handle>& handles = m_seenObjects.getObjects();
			for( int i = 1/*skip null*/; i < handles.getSize(); ++i )
			{
				hkDataObject obj = world->findObject(handles[i]);
				writeObject(obj, os);
			}

			m_indenter.pop();
			os.printf("\n</hktagfile>\n");
		}
		
		Indenter m_indenter;
		SeenObjects m_seenObjects;
		hkBool m_writeFloatAsHex;
		hkBool m_floatComment;			///< If floatAsHex is enabled, if this is enabled a comment will appear with every float holding human readable version
	};
}

hkResult hkXmlTagfileWriter::save( const hkDataObject& obj, hkStreamWriter* stream, hkTagfileWriter::AddDataObjectListener* userListener, const Options& options)
{
	Context context(userListener, options);
	context.setContents(obj);
 	context.writeFile(obj, stream);
	stream->flush();
	
	return stream->isOk() ? HK_SUCCESS : HK_FAILURE;
}

// explicitly instantiate our map type
#include <Common/Base/Container/PointerMap/hkMap.cxx>
template class hkMapBase<hkDataObject::Handle, int, HandleOps>;
template class hkMap<hkDataObject::Handle, int, HandleOps>;

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
