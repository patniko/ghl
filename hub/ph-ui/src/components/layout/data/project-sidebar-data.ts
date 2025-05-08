import {
  IconArrowLeft,
  IconChecklist,
  IconDatabase,
  IconReportAnalytics,
  IconTool,
} from '@tabler/icons-react'
import { type SidebarData } from '../types'

export const projectSidebarData: SidebarData = {
  user: {
    name: 'satnaing',
    email: 'satnaingdev@gmail.com',
    avatar: '/avatars/shadcn.jpg',
  },
  teams: [
    {
      name: 'Shadcn Admin',
      logo: () => null,
      plan: 'Vite + ShadcnUI',
    },
  ],
  navGroups: [
    {
      title: 'Navigation',
      items: [
        {
          title: 'Back to Dashboard',
          url: '/$orgSlug',
          icon: IconArrowLeft,
          dynamic: true
        },
      ],
    },
    {
      title: 'Dataset',
      items: [
        {
          title: 'Config',
          url: '/$orgSlug/$projectName/config',
          icon: IconTool,
          dynamic: true
        },
        {
          title: 'Batches',
          url: '/$orgSlug/$projectName/batches',
          icon: IconDatabase,
          dynamic: true
        },
        {
          title: 'Results',
          url: '/$orgSlug/$projectName/results',
          icon: IconReportAnalytics,
          dynamic: true
        },
        {
          title: 'Tasks',
          url: '/$orgSlug/$projectName/tasks',
          icon: IconChecklist,
          dynamic: true
        },
      ],
    },
    /*{
      title: 'Other',
      items: [
        {
          title: 'Settings',
          url: '/settings',
          icon: IconSettings,
        },
        {
          title: 'Help Center',
          url: '/help-center',
          icon: IconHelp,
        },
      ],
    },*/
  ],
}
