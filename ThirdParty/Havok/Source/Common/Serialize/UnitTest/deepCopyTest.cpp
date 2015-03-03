/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Reflection/Registry/hkTypeInfoRegistry.h>
#include <Common/Base/Types/Geometry/LocalFrame/hkLocalFrame.h>
#include <Common/Serialize/Copier/hkDeepCopier.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>

static int DeepCopyTest()
{
	void* copy;
	{
		struct HK_VISIBILITY_HIDDEN CountNumCopies : public hkTypeInfoRegistry
		{
			int m_count;
			CountNumCopies() : m_count(0) {}
			virtual const hkTypeInfo* finishLoadedObject( void* obj, const char* className ) const
			{
				const_cast<CountNumCopies*>(this)->m_count++;
				return HK_NULL;
			}
		};
		CountNumCopies countNumCopies;
		hkSimpleLocalFrame parent;
		hkSimpleLocalFrame child0;
		hkSimpleLocalFrame child1;
		parent.m_children.pushBack(&child0); child0.addReference();
		parent.m_children.pushBack(&child1); child1.addReference();
		child0.m_children.pushBack(&child1); child1.addReference();
		child1.m_children.pushBack(&child0); child0.addReference();
		copy = hkDeepCopier::deepCopy( &parent, hkSimpleLocalFrameClass, HK_NULL
			, hkBuiltinTypeRegistry::getInstance().getVtableClassRegistry()
			, &countNumCopies );
		HK_TEST(countNumCopies.m_count == 3);
	}
	{
		hkSimpleLocalFrame* parent = static_cast<hkSimpleLocalFrame*>(copy);
		hkSimpleLocalFrame* child0 = static_cast<hkSimpleLocalFrame*>(parent->m_children[0]);
		hkSimpleLocalFrame* child1 = static_cast<hkSimpleLocalFrame*>(parent->m_children[1]);
		HK_TEST( child0 != child1 );
		HK_TEST( child0->m_children[0] == child1 );
		HK_TEST( child1->m_children[0] == child0 );
	}
	hkDeepCopier::freeDeepCopy(copy);

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(DeepCopyTest, "Fast", "Common/Test/UnitTest/Serialize/", __FILE__     );

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
