/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Data/hkDataObject.h>

#include <Common/Base/Container/String/hkStringBuf.h>

//////////////////////////////////////////////////////////////////////////
// Forwarding Class Methods
//////////////////////////////////////////////////////////////////////////

void hkDataClass::getAllDeclaredMemberInfo( hkArrayBase<hkDataClass::MemberInfo>& minfos ) const
{
	// todo virtual method in impl
	int n = getNumDeclaredMembers();
	HK_ASSERT2(0x126c746e, minfos.getSize() == n, "Resize the array before calling");
	for( int i = 0; i < n; ++i )
	{
		m_impl->getDeclaredMemberInfo(i, minfos[i]);
	}
}

void hkDataClass::getCinfo( hkDataClass::Cinfo& cinfo ) const
{
	cinfo.name = m_impl->getName();
	cinfo.version = m_impl->getVersion();
	cinfo.parent = m_impl->getParent() ? m_impl->getParent()->getName() : HK_NULL;

	int nmembers = m_impl->getNumDeclaredMembers();
	cinfo.members.setSize(nmembers);

	for( int i = 0; i < nmembers; ++i )
	{
		hkDataClass::MemberInfo msrc;
		m_impl->getDeclaredMemberInfo(i, msrc);

		hkDataClass::Cinfo::Member& mdst = cinfo.members[i];
		mdst.set( msrc.m_name, msrc.m_type);
	}
}


//////////////////////////////////////////////////////////////////////////
// Forwarding Object Methods
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// Object Value Methods
//////////////////////////////////////////////////////////////////////////

hkDataObject::Type hkDataObject::Value::getType() const
{
	hkDataClass::MemberInfo minfo;
	m_impl->getMemberInfo(m_handle, minfo);
	return minfo.m_type;
}

//////////////////////////////////////////////////////////////////////////
// Array Methods
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// Array Value Methods
//////////////////////////////////////////////////////////////////////////

void hkDataWorld::renameClass(hkDataClass& klass, const char* newName) { HK_ASSERT(0x1b059d46,0); }
void hkDataWorld::removeClass(hkDataClass& klass) { HK_ASSERT(0x2f609896,0); }

void hkDataWorld::setClassVersion(hkDataClass& klass, int newVersion) { HK_ASSERT(0x6f66aebe,0); }
void hkDataWorld::setClassParent(hkDataClass& klass, hkDataClass& parent) { HK_ASSERT(0x4dbbc51,0); }

void hkDataWorld::renameClassMember(hkDataClass& klass, const char* oldName, const char* newName) { HK_ASSERT(0x38485262,0); }
void hkDataWorld::removeClassMember(hkDataClass& klass, const char* name) { HK_ASSERT(0x6da37bd9,0); }
void hkDataWorld::findObjectsByBaseClass(const char* baseClassName, hkArray<hkDataObjectImpl*>::Temp& objectsOut) const { HK_ASSERT(0x5a205cd7,0); }
void hkDataWorld::findObjectsByExactClass(const char* className, hkArray<hkDataObjectImpl*>::Temp& objectsOut) const { HK_ASSERT(0x314641bd,0); }
void hkDataWorld::castObject(hkDataObject& obj, const hkDataClass& objClass) { HK_ASSERT(0x6b4d8e26,0); }
void hkDataWorld::setClassMemberDefault(hkDataClass& klass, const char* name, const void* valuePtr) { HK_ASSERT(0x474835f6, 0);}

hkDataClassImpl* hkDataWorld::copyClassFromWorld(const char* name, const hkDataWorld& worldFrom) 
{
	return HK_NULL;
}

void hkDataWorld::addClassMember(hkDataClass& klass, const char* name, hkDataObject::Type type, const void* valuePtr)
{
	HK_ASSERT(0x474835f6, 0);
}

void hkDataWorld::addClassMemberTypeExpression(hkDataClass& klass, const char* name, const char* expr, const char* className, const void* valuePtr)
{
	hkTypeManager& typeManager = getTypeManager();

	hkTypeManager::Type* type = HK_NULL;

	if (className)
	{
		hkStringBuf work;
		work.append(expr);
		work.append("C");
		work.append(className);
		work.append(";");

		type = typeManager.parseTypeExpression(work);	
	}
	else
	{
		type = typeManager.parseTypeExpression(expr);
	}

	HK_ASSERT(0x2342423, type);

	if (type)
	{
		addClassMember(klass, name, type, valuePtr);
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
