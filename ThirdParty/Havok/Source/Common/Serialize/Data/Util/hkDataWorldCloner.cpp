/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Data/Util/hkDataWorldCloner.h>
#include <Common/Serialize/Data/hkDataObject.h>

hkDataWorldCloner::hkDataWorldCloner( hkDataWorld& dstWorld, const hkDataWorld& srcWorld )
	: m_dstWorld(dstWorld), m_srcWorld(srcWorld)
{
	HK_ASSERT(0x48e8992d, m_dstWorld.getContents().isNull() );
}

template<typename Value>
void hkDataWorldCloner::copySimpleValue(Value dst, const Value& src )
{
	hkDataObject::Type srcType = src.getType();

	switch( srcType->getSubType() )
	{
		case hkTypeManager::SUB_TYPE_VOID:
		{
			break;
		}
		case hkTypeManager::SUB_TYPE_BYTE:
		case hkTypeManager::SUB_TYPE_INT:
		{
			dst = src.asInt();
			break;
		}
		case hkTypeManager::SUB_TYPE_REAL:
		{
			dst = src.asReal();
			break;
		}
		case hkTypeManager::SUB_TYPE_TUPLE:
		{
			if (srcType->getParent()->isReal())
			{
				dst.setVec( src.asVec(srcType->getTupleSize()), srcType->getTupleSize() );
			}
			else
			{
				HK_ASSERT(0x213abbc, !"Can't copy other tuple types");
			}
			break;
		}
		case hkTypeManager::SUB_TYPE_POINTER:
		case hkTypeManager::SUB_TYPE_CLASS:
		{
			dst = copyObject( src.asObject() );
			break;
		}
		case hkTypeManager::SUB_TYPE_CSTRING:
		{
			dst = src.asString();
			break;
		}
		default:
		{
			HK_ASSERT(0x51a991bd, 0);
		}
	}
}

void hkDataWorldCloner::copyArray( hkDataArray& dst, const hkDataArray& src, int arraySize)
{
	hkDataObject::Type type = dst.getType();
	switch( type->getSubType())
	{
		case hkTypeManager::SUB_TYPE_VOID:
		{
			break;
		}
		case hkTypeManager::SUB_TYPE_BYTE:
		case hkTypeManager::SUB_TYPE_INT:
		{
			for( int i = 0; i < arraySize; ++i )
			{
				dst[i] = src[i].asInt();
			}
			break;
		}
		case hkTypeManager::SUB_TYPE_REAL:
		{
			for( int i = 0; i < arraySize; ++i )
			{
				dst[i] = src[i].asReal();
			}
			break;
		}
		case hkTypeManager::SUB_TYPE_TUPLE:
		{
			const int size = type->getTupleSize();
			if (type->getParent()->getSubType() == hkTypeManager::SUB_TYPE_REAL)
			{
				for( int i = 0; i < arraySize; ++i )
				{
					dst[i].setVec( src[i].asVec(size), size );
				}
			}
			else
			{
				dst.setSize(arraySize);
				HK_ASSERT(0x243242a3, src.getSize() == arraySize);

				// Try to copy other tuple types - by just doing an array copy

				for (int i = 0; i < arraySize; i++)
				{	
					hkDataArray srcChild = src[i].asArray();
					hkDataArray dstChild = dst[i].asArray();

					HK_ASSERT(0x2432423, srcChild.getSize() == size);
					dstChild.setSize(size);

					copyArray(dstChild, srcChild, size);
				}
			}
			break;
		}
		case hkTypeManager::SUB_TYPE_ARRAY:
		{
			dst.setSize(arraySize);
			for (int i = 0; i < arraySize; i++)
			{
				hkDataArray da = dst[i].asArray();
				hkDataArray sa = src[i].asArray();

				// Copy the arrays over

				da.setSize(sa.getSize());
				copyArray(da, sa, sa.getSize());
			}
			break;
		}
		case hkTypeManager::SUB_TYPE_POINTER:
		{
			dst.setSize(arraySize);
			for( int i = 0; i < arraySize; ++i )
			{
				dst[i] = copyObject(src[i].asObject());
			}
			break;
		}
		case hkTypeManager::SUB_TYPE_CLASS:
		{
			dst.setSize( arraySize );
			hkDataClass srcClass = src.getClass();
			int numMembers = srcClass.getNumMembers();
			hkArray<hkDataClass::MemberInfo>::Temp minfos(numMembers);
			minfos.setSizeUnchecked(numMembers);
			srcClass.getAllMemberInfo(minfos);
			for( int memberIndex = 0; memberIndex < numMembers; ++memberIndex )
			{
				const hkDataClass::MemberInfo& mi = minfos[memberIndex];
				if( mi.m_type->isArray() || (mi.m_type->isTuple() && !mi.m_type->getParent()->isReal()))
				{
					hkDataArray da = dst.swizzleObjectMember(mi.m_name);
					hkDataArray sa = src.swizzleObjectMember(mi.m_name);

					da.setSize(sa.getSize());

					copyArray(da, sa, sa.getSize());

#if 0
					for( int ai = 0; ai < arraySize; ++ai )
					{
						hkDataArray dt = da[ai].asArray();
						hkDataArray st = sa[ai].asArray();
						HK_ASSERT(0x25087f44, dt.getSize() == st.getSize() );
						int tupleCount = st.getSize();
						for( int ti = 0; ti < tupleCount; ++ti )
						{
							copySimpleValue( dt[ti], st[ti] );
						}
					}
#endif
				}
				else if( !mi.m_type->isVoid())
				{
					hkDataArray d = dst.swizzleObjectMember(mi.m_name);
					hkDataArray s = src.swizzleObjectMember(mi.m_name);
					for( int i = 0; i < arraySize; ++i )
					{
						copySimpleValue( d[i], s[i] );
					}
				}
			}
			break;
		}
		case hkTypeManager::SUB_TYPE_CSTRING:
		{
			for( int i = 0; i < arraySize; ++i )
			{
				dst[i] = src[i].asString();
			}
			break;
		}
		default:
		{
			HK_ASSERT(0x51a991bd, 0);
		}
	}
}

void hkDataWorldCloner::copyValue( hkDataObject& dstObj, const hkDataObject& srcObj, const hkDataClass::MemberInfo& minfo )//::Value srcVal, hkDataObject::Value dst)
{
	hkDataObject::Type type = minfo.m_type;
	if( type->isTuple() && !type->getParent()->isReal())
	{
		hkDataArray src = srcObj[minfo.m_name].asArray();
		hkDataArray dst = dstObj[minfo.m_name].asArray();
		dst.setSize( minfo.m_type->getTupleSize());
		copyArray( dst, src, minfo.m_type->getTupleSize() );
	}
	else if( type->isArray())
	{
		hkDataArray src = srcObj[minfo.m_name].asArray();
		hkDataArray dst = dstObj[minfo.m_name].asArray();
		dst.setSize(src.getSize());
		copyArray( dst, src, src.getSize() );
	}
	else
	{
		hkDataObject::Value d = dstObj[minfo.m_name];
		copySimpleValue( d, srcObj[minfo.m_name] );
	}
}

hkDataClassImpl* hkDataWorldCloner::findDestClass(const char* classname)
{
	if( hkDataClassImpl* c = m_dstWorld.findClass(classname) )
	{
		return c;
	}

	hkArray<hkDataClass::Cinfo>::Temp cinfos; cinfos.reserve(10);
	{
		const char* cname = classname;
		while( cname && m_dstWorld.findClass(cname) == HK_NULL )
		{
			hkDataClass sc = m_srcWorld.findClass(cname);
			sc.getCinfo( cinfos.expandOne() );
			cname = sc.getParent().isNull() ? HK_NULL : sc.getParent().getName();
		}
	}
	for( int cIndex = cinfos.getSize()-1; cIndex >= 0; --cIndex )
	{
		hkDataClass::Cinfo dstInfo;
		
		const hkDataClass::Cinfo& srcInfo = cinfos[cIndex];

		dstInfo.name = srcInfo.name;
		dstInfo.version = srcInfo.version;
		dstInfo.parent = srcInfo.parent;

		const int numMembers = srcInfo.members.getSize();
		dstInfo.members.setSize(numMembers);
		for (int i = 0; i < numMembers; i++)
		{
			const hkDataClass::Cinfo::Member& srcMem = srcInfo.members[i];
			hkDataClass::Cinfo::Member& dstMem = dstInfo.members[i];
		
			dstMem = srcMem;
			dstMem.type = m_dstWorld.getTypeManager().copyType(srcMem.type);
		}

		m_dstWorld.newClass( dstInfo );
	}
	
	{
		hkDataClassImpl* csrc = m_srcWorld.findClass(classname);
		for( int i = 0; i < csrc->getNumMembers(); ++i )
		{
			hkDataClass::MemberInfo minfo;
			csrc->getMemberInfo(i, minfo);

			hkDataObject::Type term = minfo.m_type->findTerminal();

			if( term->isClass() )
			{
				findDestClass( term->getTypeName());
			}
		}
	}
	return m_dstWorld.findClass(classname);
}

void hkDataWorldCloner::getClassMemberInfos( hkArray<hkDataClass::MemberInfo>::Temp& dstInfos, const hkDataClass& dstClass, const hkDataClass& srcClass )
{
	HK_ASSERT(0x7d8b0fe4, srcClass.getNumMembers() == dstClass.getNumMembers());
	dstInfos.setSize(dstClass.getNumMembers());
	dstClass.getAllMemberInfo(dstInfos);

	hkArray<hkDataClass::MemberInfo>::Temp srcInfos;
	srcInfos.setSize(srcClass.getNumMembers());
	srcClass.getAllMemberInfo(srcInfos);

	for( int i = 0; i < srcClass.getNumMembers(); ++i )
	{
		hkDataObject::Type term = srcInfos[i].m_type->findTerminal();

		if (term->isClass())
		{
			findDestClass(term->getTypeName());
		}
	}
}

void hkDataWorldCloner::copyObjectMembers( hkDataObject& dstObject, const hkDataObject& srcObject )
{
	hkDataClass dstClass = dstObject.getClass();
	hkArray<hkDataClass::MemberInfo>::Temp minfos;
	getClassMemberInfos( minfos, dstClass, srcObject.getClass() );		
	for( int memIndex = 0; memIndex < minfos.getSize(); ++memIndex )
	{
		copyValue(dstObject, srcObject, minfos[memIndex]);
	}
}

hkDataObject hkDataWorldCloner::copyObject( const hkDataObject& srcObject )
{
	if( srcObject.isNull() == hkFalse32 )
	{
		hkDataObject::Handle dstHandle;
		if( m_copied.get(srcObject.getHandle(), &dstHandle) == HK_SUCCESS )
		{
			return m_dstWorld.findObject(dstHandle);
		}
		else
		{
			hkDataClass dstClass = findDestClass(srcObject.getClass().getName());
			HK_ASSERT(0x2518e100, dstClass.isNull() == hkFalse32 );
			hkDataObject dstObject = m_dstWorld.newObject(dstClass);
			m_copied.insert( srcObject.getHandle(), dstObject.getHandle() );
			copyObjectMembers( dstObject, srcObject );
			return dstObject;
		}
	}
	return HK_NULL;
}

hkResult hkDataWorldCloner::clone()
{
	copyObject( m_srcWorld.getContents() );
	return HK_SUCCESS;
}

// explicitly instantiate our map type
#include <Common/Base/Container/PointerMap/hkMap.cxx>
template class hkMapBase<hkDataObject::Handle, int>;
template class hkMap<hkDataObject::Handle, int>;

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
