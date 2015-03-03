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

static int dataObject()
{
	{
		hkDataWorldDict world;
		hkTypeManager& typeManager = world.getTypeManager();

		const char* class1Name = "Foo1";
		const char* class2Name = "Foo2";
		const char* m_pointer_Foo1 = "pointer_Foo1";
		const char* m_int = "int";
		const char* m_renamed_int = "renamed_int";
		const char* m_array_struct_Foo1 = "array_struct_Foo1";
		const char* m_tuple_struct_Foo1 = "tuple_struct_Foo1";
		const char* m_tuple_int = "tuple_int";
		const char* m_renamed_tuple_int = "renamed_tuple_int";

		{
			// create class Foo1
			{
				hkDataClass::Cinfo cinfo;
				cinfo.name = class1Name;
				cinfo.version = 0;
				cinfo.parent = HK_NULL;
				HK_TEST(world.findClass(cinfo.name) == HK_NULL);
				world.newClass(cinfo);
				HK_TEST(world.findClass(cinfo.name) != HK_NULL);
			}
			HK_ASSERT(0x66a67906, world.findClass(class1Name));
			hkDataClass foo1class(world.findClass(class1Name));
			world.addClassMember(foo1class, m_pointer_Foo1, typeManager.getClassPointer(class1Name), HK_NULL);
			HK_TEST(foo1class.getDeclaredMemberIndexByName(m_pointer_Foo1) != -1);
			world.addClassMember(foo1class, m_int, typeManager.getSubType(hkTypeManager::SUB_TYPE_INT), HK_NULL);
			HK_TEST(foo1class.getDeclaredMemberIndexByName(m_int) != -1);
			world.addClassMember(foo1class, m_tuple_int, typeManager.parseTypeExpression("{3}i"), HK_NULL);
			HK_TEST(foo1class.getDeclaredMemberIndexByName(m_tuple_int) != -1);

			// create class Foo2
			{
				hkDataClass::Cinfo cinfo;
				cinfo.name = class2Name;
				cinfo.version = 0;
				cinfo.parent = HK_NULL;
				HK_TEST(world.findClass(cinfo.name) == HK_NULL);
				world.newClass(cinfo);
				HK_TEST(world.findClass(cinfo.name) != HK_NULL);
			}
			HK_ASSERT(0x66a67906, world.findClass(class2Name));
			hkDataClass foo2class(world.findClass(class2Name));
			world.addClassMember(foo2class, m_array_struct_Foo1, typeManager.makeArray(typeManager.addClass(foo1class.getName())), HK_NULL);
			HK_TEST(foo2class.getDeclaredMemberIndexByName(m_array_struct_Foo1) != -1);
			world.addClassMember(foo2class, m_int, typeManager.parseTypeExpression("i"), HK_NULL);
			HK_TEST(foo2class.getDeclaredMemberIndexByName(m_int) != -1);
			world.addClassMember(foo2class, m_tuple_int, typeManager.parseTypeExpression("{4}i"), HK_NULL);
			HK_TEST(foo2class.getDeclaredMemberIndexByName(m_tuple_int) != -1);
			world.addClassMember(foo2class, m_tuple_struct_Foo1, typeManager.makeTuple(typeManager.addClass(foo1class.getName()), 2), HK_NULL);
			HK_TEST(foo2class.getDeclaredMemberIndexByName(m_array_struct_Foo1) != -1);

			hkArray<hkDataObjectImpl*>::Temp objects;
			{
				// create objects
				HK_TEST(world.getContents().isNull());
				world.findObjectsByBaseClass(foo1class.getName(), objects);
				HK_TEST(objects.getSize() == 0);
				// Foo1
				hkDataObject obj1 = world.newObject(foo1class);
				HK_TEST(!obj1.isNull());
				world.findObjectsByBaseClass(foo1class.getName(), objects);
				HK_TEST(objects.getSize() == 1);
				hkDataObject obj2 = world.newObject(foo1class);
				HK_TEST(!obj2.isNull());
				world.findObjectsByBaseClass(foo1class.getName(), objects);
				HK_TEST(objects.getSize() == 2);

				// obj1
				// - object member
				HK_TEST(!obj1.hasMember(m_pointer_Foo1));
				HK_TEST(obj1[m_pointer_Foo1].asObject().isNull());
				obj1[m_pointer_Foo1] = obj2;
				HK_TEST(obj1.hasMember(m_pointer_Foo1));
				HK_TEST(!obj1[m_pointer_Foo1].asObject().isNull());
				// - int member
				HK_TEST(!obj1.hasMember(m_int));
				obj1[m_int] = 1;
				HK_TEST(obj1.hasMember(m_int));
				// - tuple of ints member
				HK_TEST(!obj1.hasMember(m_tuple_int));
				obj1[m_tuple_int].asArray()[0] = 1;
				obj1[m_tuple_int].asArray()[2] = -1;
				HK_TEST(obj1[m_tuple_int].asArray()[0].asInt() == 1);
				HK_TEST(obj1[m_tuple_int].asArray()[1].asInt() == 0);
				HK_TEST(obj1[m_tuple_int].asArray()[2].asInt() == -1);

				// obj2
				// - object member
				HK_TEST(!obj2.hasMember(m_pointer_Foo1));
				HK_TEST(obj2[m_pointer_Foo1].asObject().isNull());
				obj2[m_pointer_Foo1] = obj1;
				HK_TEST(obj2.hasMember(m_pointer_Foo1));
				HK_TEST(!obj2[m_pointer_Foo1].asObject().isNull());
				// - int member
				HK_TEST(!obj2.hasMember(m_int));
				obj2[m_int] = 2;
				HK_TEST(obj2.hasMember(m_int));
				HK_TEST(obj2[m_int].asInt() == 2);
				// - tuple of ints member
				HK_TEST(obj1[m_pointer_Foo1].asObject()[m_int].asInt() == 2);
				HK_TEST(obj2[m_pointer_Foo1].asObject()[m_int].asInt() == 1);

				// world contents
				HK_TEST(!world.getContents().isNull());
				HK_TEST(world.getContents()[m_int].asInt() == 1); // obj1, first created object

				// Foo2
				hkDataObject obj3 = world.newObject(foo2class);
				HK_TEST(!obj3.isNull());
				world.findObjectsByBaseClass(foo2class.getName(), objects);
				HK_TEST(objects.getSize() == 1);

				// obj3
				// array of objects member
				HK_TEST(!obj3.hasMember(m_array_struct_Foo1));
				HK_TEST(obj3[m_array_struct_Foo1].asArray().getSize() == 0);
				HK_TEST(obj3.hasMember(m_array_struct_Foo1));
				obj3[m_array_struct_Foo1].asArray().setSize(2);
				obj3[m_array_struct_Foo1].asArray()[0] = obj1;
				obj3[m_array_struct_Foo1].asArray()[1] = obj2;
				obj3[m_array_struct_Foo1].asArray()[0].asObject()[m_int] = 100;
				HK_TEST(obj3[m_array_struct_Foo1].asArray()[0].asObject()[m_int].asInt() != obj1[m_int].asInt());
				obj3[m_array_struct_Foo1].asArray()[0].asObject()[m_tuple_int].asArray()[1] = -2;
				HK_TEST(obj3[m_array_struct_Foo1].asArray()[0].asObject()[m_tuple_int].asArray()[1].asInt() != obj1[m_tuple_int].asArray()[1].asInt());
				// int member
				HK_TEST(!obj3.hasMember(m_int));
				obj3[m_int] = 3;
				HK_TEST(obj3[m_int].asInt() == 3);
				HK_TEST(obj3.hasMember(m_int));
				// tuple of ints member
				HK_TEST(!obj3.hasMember(m_tuple_int));
				obj3[m_tuple_int].asArray()[0] = 1;
				obj3[m_tuple_int].asArray()[2] = -1;
				HK_TEST(obj3.hasMember(m_tuple_int));
				HK_TEST(obj3[m_tuple_int].asArray().getSize() == 4);
				HK_TEST(obj3[m_tuple_int].asArray()[0].asInt() == 1);
				HK_TEST(obj3[m_tuple_int].asArray()[1].asInt() == 0);
				HK_TEST(obj3[m_tuple_int].asArray()[2].asInt() == -1);
				HK_TEST(obj3[m_tuple_int].asArray()[3].asInt() == 0);
				// tuple of structs member
				HK_TEST(!obj3.hasMember(m_tuple_struct_Foo1));
				obj3[m_tuple_struct_Foo1].asArray()[0] = obj1;
				obj3[m_tuple_struct_Foo1].asArray()[1] = obj2;
				HK_TEST(obj3.hasMember(m_tuple_struct_Foo1));
				HK_TEST(obj3[m_tuple_struct_Foo1].asArray().getSize() == 2);
				HK_TEST(obj3[m_tuple_struct_Foo1].asArray()[0].asObject()[m_int].asInt() == obj1[m_int].asInt());
				obj3[m_tuple_struct_Foo1].asArray()[0].asObject()[m_int] = 5;
				HK_TEST(obj3[m_tuple_struct_Foo1].asArray()[0].asObject()[m_int].asInt() != obj1[m_int].asInt());
				obj3[m_tuple_struct_Foo1].asArray()[0].asObject()[m_tuple_int].asArray()[1] = 3;
				HK_TEST(obj3[m_tuple_struct_Foo1].asArray()[0].asObject()[m_tuple_int].asArray()[1].asInt() != obj1[m_tuple_int].asArray()[1].asInt());

				// rename foo1class class members
				// - int
				HK_TEST(foo1class.getDeclaredMemberIndexByName(m_renamed_int) == -1);
				world.renameClassMember(foo1class, m_int, m_renamed_int);
				HK_TEST(foo1class.getDeclaredMemberIndexByName(m_renamed_int) != -1);
				HK_TEST(foo1class.getDeclaredMemberIndexByName(m_int) == -1);
				HK_TEST(obj1[m_renamed_int].asInt() == 1);
				HK_TEST(obj2[m_renamed_int].asInt() == 2);
				HK_TEST(obj3[m_array_struct_Foo1].asArray()[0].asObject()[m_renamed_int].asInt() == 100); // obj1 copy
				HK_TEST(obj3[m_array_struct_Foo1].asArray()[1].asObject()[m_renamed_int].asInt() == 2); // obj2 copy
				HK_TEST(obj3[m_tuple_struct_Foo1].asArray()[0].asObject()[m_renamed_int].asInt() == 5); // obj1 copy
				HK_TEST(obj3[m_tuple_struct_Foo1].asArray()[1].asObject()[m_renamed_int].asInt() == 2); // obj2 copy
				// - tuple of ints
				HK_TEST(foo1class.getDeclaredMemberIndexByName(m_renamed_tuple_int) == -1);
				world.renameClassMember(foo1class, m_tuple_int, m_renamed_tuple_int);
				HK_TEST(foo1class.getDeclaredMemberIndexByName(m_renamed_tuple_int) != -1);
				HK_TEST(foo1class.getDeclaredMemberIndexByName(m_tuple_int) == -1);
				HK_TEST(obj1[m_renamed_tuple_int].asArray()[0].asInt() == 1);
				HK_TEST(obj1[m_renamed_tuple_int].asArray()[1].asInt() == 0);
				HK_TEST(obj1[m_renamed_tuple_int].asArray()[2].asInt() == -1);
				HK_TEST(obj3[m_array_struct_Foo1].asArray()[0].asObject()[m_renamed_tuple_int].asArray()[0].asInt() == 1); // obj1 copy
				HK_TEST(obj3[m_array_struct_Foo1].asArray()[0].asObject()[m_renamed_tuple_int].asArray()[1].asInt() == -2); // obj1 copy
				HK_TEST(obj3[m_array_struct_Foo1].asArray()[0].asObject()[m_renamed_tuple_int].asArray()[2].asInt() == -1); // obj1 copy
				HK_TEST(obj3[m_tuple_struct_Foo1].asArray()[0].asObject()[m_renamed_tuple_int].asArray()[0].asInt() == 1); // obj1 copy
				HK_TEST(obj3[m_tuple_struct_Foo1].asArray()[0].asObject()[m_renamed_tuple_int].asArray()[1].asInt() == 3); // obj1 copy
				HK_TEST(obj3[m_tuple_struct_Foo1].asArray()[0].asObject()[m_renamed_tuple_int].asArray()[2].asInt() == -1); // obj1 copy
			}
			//
			// remove the class (should remove all foo1class objects)
			//

			// remove members of the foo1class type first
			world.removeClassMember(foo1class, m_pointer_Foo1);
			world.removeClassMember(foo2class, m_array_struct_Foo1);
			world.removeClassMember(foo2class, m_tuple_struct_Foo1);
			// remove foo1class
			world.removeClass(foo1class);
			HK_TEST(world.findClass(class1Name) == HK_NULL);
			world.findObjectsByBaseClass(class1Name, objects);
			HK_TEST(objects.getSize() == 0);
			HK_TEST(world.getContents().isNull());
		}
	}

	{
		hkDataWorldDict world2;
		hkTypeManager& typeManager = world2.getTypeManager();
		hkDataClass::Cinfo cinfo;
		cinfo.name = "Empty";
		cinfo.parent = HK_NULL;
		cinfo.version = 0;
		hkDataClass emptyClass( world2.newClass(cinfo) );

		cinfo.name = "Class0";
		cinfo.members.expandOne().set("int0", typeManager.getSubType(hkTypeManager::SUB_TYPE_INT));
		cinfo.members.expandOne().set("obj0", typeManager.getClassPointer("Empty"));
		hkDataClass class0( world2.newClass(cinfo) );

		cinfo.name = "Class1";
		cinfo.members[0].name = "int1";
		cinfo.members[1].name = "obj1";
		hkDataClass class1( world2.newClass(cinfo) );

		hkDataObject empty = world2.newObject(emptyClass);
		hkDataObject obj0 = world2.newObject(class0);
		hkDataObject obj1 = world2.newObject(class1);

		obj0["int0"] = 10;
		obj1["int1"] = obj0["int0"];
		obj0["obj0"] = empty;
		obj1["obj1"] = obj0["obj0"];
		//obj1["obj1"] = 20; // should assert incompatible types
		//obj1["int1"] = empty; // should assert incompatible types
	}
	
	return 0;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(dataObject, "Fast", "Common/Test/UnitTest/Serialize/", __FILE__     );

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
