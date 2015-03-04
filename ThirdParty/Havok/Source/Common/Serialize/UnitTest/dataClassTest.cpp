/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Serialize/Data/Dict/hkDataObjectDict.h>
#include <Common/Serialize/Data/hkDataObjectImpl.h>

static hkBool32 cinfoMatches( const hkDataClass& k, const hkDataClass::Cinfo& c0)
{
	hkDataClass::Cinfo c1; k.getCinfo(c1);
	if( hkString::strCmp(c0.name, c1.name) != 0 ) return false;
	if( c0.version != c1.version )  return false;
	if( (c0.parent==HK_NULL) != (c1.parent==HK_NULL) ) return false;
	if( c0.parent && hkString::strCmp(c0.parent, c1.parent) != 0 ) return false;
	if( c0.members.getSize() != c1.members.getSize() ) return false;
	for( int i = 0; i < c0.members.getSize(); ++i )
	{
		const hkDataClass::Cinfo::Member& m0 = c0.members[i];
		const hkDataClass::Cinfo::Member& m1 = c1.members[i];
		if ( hkString::strCmp(m0.name, m1.name) != 0 ) return false;
		if (!m0.type->isEqual(m1.type)) return false;
	}
	return true;
}

static int dataClass()
{
	hkDataWorldDict world;

	const char* className1 = "Foo1";
	const char* className2 = "Foo2";
	const char* memberName = "memberFoo1";

	// add class
	{
		hkDataClass::Cinfo cinfo;
		cinfo.name = className1;
		cinfo.version = 0;
		cinfo.parent = HK_NULL;
		HK_TEST(world.findClass(cinfo.name) == HK_NULL);
		hkDataClass knew = world.newClass(cinfo);
		HK_TEST( cinfoMatches(knew, cinfo) );
		HK_TEST(world.findClass(cinfo.name) != HK_NULL);
		HK_ASSERT(0x6a34df2d, world.findClass(cinfo.name));
	}
	// add class with member referencing another class
	{
		hkDataClass::Cinfo cinfo;
		cinfo.name = className2;
		cinfo.version = 0;
		cinfo.parent = HK_NULL;
		HK_TEST(world.findClass(cinfo.name) == HK_NULL);
		hkDataClass knew = world.newClass(cinfo);
		HK_TEST( cinfoMatches(knew, cinfo) );

		hkDataClassImpl* foo2 = world.findClass(cinfo.name);
		HK_TEST(foo2 != HK_NULL);
		HK_ASSERT(0x3d527e8b, foo2);
		HK_TEST(foo2->getDeclaredMemberIndexByName(memberName) == -1);

		hkDataClass foo2class(foo2);
		world.addClassMemberTypeExpression(foo2class, memberName, "*", className1, HK_NULL);
		HK_TEST(foo2->getDeclaredMemberIndexByName(memberName) != -1);
		HK_ASSERT(0x7b7dfa61, foo2->getDeclaredMemberIndexByName(memberName) != -1);
	}
	// rename referenced class and check member
	{
		hkDataClassImpl* foo1 = world.findClass(className1);
		hkDataClassImpl* foo2 = world.findClass(className2);
		int mindex = foo2->getDeclaredMemberIndexByName(memberName);
		hkDataClass::MemberInfo minfo;
		foo2->getDeclaredMemberInfo(mindex, minfo);

		HK_TEST(minfo.m_type->findTerminal()->isClass());
		hkDataClass foo1class(foo1);
		world.renameClass(foo1class, "Renamed_Foo1");
		foo2->getDeclaredMemberInfo(mindex, minfo);
		HK_TEST(minfo.m_type->findTerminal()->isClass());
		world.renameClass(foo1class, className1);
		foo2->getDeclaredMemberInfo(mindex, minfo);
		HK_TEST(minfo.m_type->findTerminal()->isClass());
	}
	// set parent
	{
		hkDataClass foo1class(world.findClass(className1));
		hkDataClass foo2class(world.findClass(className2));
		HK_TEST(foo2class.getParent().isNull());
		world.setClassParent(foo2class, foo1class);
		HK_TEST(!foo2class.getParent().isNull());
		HK_ASSERT(0x698d3f45, !foo2class.getParent().isNull());
		HK_TEST(hkString::strCmp(foo2class.getParent().getName(), foo1class.getName()) == 0);
	}
	// remove parent class (should remove full hierarchy)
	{
		hkDataClass foo1class(world.findClass(className1));
		HK_TEST(world.findClass(className2) != HK_NULL);
		world.removeClass(foo1class);
		HK_TEST(world.findClass(className1) == HK_NULL);
		HK_TEST(world.findClass(className2) == HK_NULL);
	}

	return 0;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(dataClass, "Fast", "Common/Test/UnitTest/Serialize/", __FILE__     );

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
